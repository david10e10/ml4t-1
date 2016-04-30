import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import KNNLearner as knn
from mc3_p2.otherfiles.marketsim import compute_portvals
from util import get_data, plot_data


def plot_orders(df, long_orders, short_orders, exit_orders, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ymin = df.min()*.9
    ymax = df.max()*1.1

    plt.vlines(long_orders.index, color='green',ymin=ymin, ymax=ymax)
    plt.vlines(short_orders.index, color='red',ymin=ymin, ymax=ymax)
    plt.vlines(exit_orders.index, color='black',ymin=ymin, ymax=ymax)

    plt.show()

def plot_corr(df1, df2, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    #ax = df.plot(title=title, fontsize=12, marker='o')
    plt.scatter(df1, df2, color = 'blue')
    plt.show()

def compute_portfolio_stats(portfolio_value, rfr = 0.0, sf = 252.0, window=20):

    daily_returns = ((portfolio_value/portfolio_value.shift(1)) - 1).ix[1:] #calculate Daily Return. Tomorrow - Today, remove the header (which is 0)

    cr = (portfolio_value.ix[-1]/portfolio_value.ix[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = np.sqrt(sf)*(daily_returns-rfr).mean()/std_daily_return

    start_date = portfolio_value.index[0].to_datetime()
    end_date = portfolio_value.index[-1].to_datetime()

    volatility = pd.rolling_std(daily_returns, window)

    # #TODO Remove troubleshooting code
    # print normalizedPrices.head()
    # print allocatedPrices.head()
    # print position_values.head()
    # print portfolio_value.head()
    # print
    # print cr
    # print avg_daily_return
    # print std_daily_return
    # print sharpe_ratio

    return cr, avg_daily_return, std_daily_return, sharpe_ratio, portfolio_value, start_date, end_date, daily_returns, volatility

def define_features(symbol, startdate_string ='12/31/07', enddate_string ='12/31/09', window=20):
    """

    :param symbol: STRING
    :param startdate_string: STRING 'MM/DD/YY'
    :param enddate_string: STRING 'MM/DD/YY'
    :param window: size of rolling averages for SMA and STDEV
    :return: data, data_np.  Features in both Pandas and Numpy formats.  4 columns each of ['bb_value', 'momentum', 'daily_returns', 'volatility']
    """
    # Import Orders into DataFrame (CURRENTLY HAS ALL DATES including non-trading)
    start_date = pd.to_datetime(startdate_string) #StartDate per Instructions
    end_date = pd.to_datetime(enddate_string) #EndDate per Instructions
    dates = pd.date_range(start_date, end_date)

    symbols = [symbol, '$SPX']

    # Read in adjusted closing prices for given symbols, date range
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[[symbol]]  # only portfolio symbols
    #prices_np = prices.as_matrix()
    #index_df = prices.index

    # Compute SMA
    sma = pd.rolling_mean(prices, window)
    sma.columns = prices.columns
    #sma_np = sma.as_matrix()

    # Compute Std Dev
    std_dev = pd.rolling_std(prices, window)
    std_dev.columns = prices.columns
    #std_dev_np = std_dev.as_matrix()

    # Calculate Normalized Bollinger Band
    bb_value = (prices-sma)/(2*std_dev)
    #bb_value_np = bb_value.as_matrix()

    # Calculate Momentum
    momentum = (prices/prices.shift(window)) - 1
    #momentum_np = momentum.as_matrix()

    # Calculate Normal Stats
    #compute_portfolio_stats(get_data(['SPY'], dates = pd.date_range(start_dates, end_dates)))
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, portfolio_value, start_dates, end_dates, daily_returns, volatility = compute_portfolio_stats(prices, window=window)


    # Combine All Features into 1 dataframe
    data = pd.concat([bb_value, momentum, daily_returns, volatility], axis = 1)
    data.columns = ['bb_value', 'momentum', 'daily_returns', 'volatility']
    data_np = data.as_matrix()
    #TODO should we chop off the NaNs?

    return data, data_np

def define_y(symbol, startdate_string ='12/31/07', enddate_string ='12/31/09', window=5):
    """

    :param symbol: STRING
    :param startdate_string: STRING 'MM/DD/YY'
    :param enddate_string: STRING 'MM/DD/YY'
    :param window: size of rolling averages for 5 day forecast
    :return: data, data_np.  Features in both Pandas and Numpy formats.  4 columns each of ['bb_value', 'momentum', 'daily_returns', 'volatility']
    """
    # Import Orders into DataFrame (CURRENTLY HAS ALL DATES including non-trading)
    start_date = pd.to_datetime(startdate_string) #StartDate per Instructions
    end_date = pd.to_datetime(enddate_string) #EndDate per Instructions
    dates = pd.date_range(start_date, end_date)

    symbols = [symbol, '$SPX']

    # Read in adjusted closing prices for given symbols, date range
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[[symbol]]  # only portfolio symbols
    #prices_np = prices.as_matrix()
    #index_df = prices.index

    # Compute SMA
    sma = pd.rolling_mean(prices, window)
    sma.columns = prices.columns
    #sma_np = sma.as_matrix()

    y = (prices.shift(-5)/prices)-1
    y_np = y.as_matrix().transpose() #need to transpose y. As a 1d Output variable, need to have it be a series.  Not sure why, but whatever

    return y, y_np, prices

def find_orders(predY, percentage):
    #Date,Symbol,Order,Shares
    dates = predY.index
    symbol = predY.columns[0]
    columns = ['Symbol', 'Order', 'Shares']
    orders = pd.DataFrame(index=dates, columns=columns)
    #orders.set_value(pd.to_datetime('12/21/09'), columns[0], 99)
    skip = -1
    intrade = False
    trade = 'long'
    for index, row in predY.iterrows():
        if not intrade:
            # if > 1%, buy and hold
            if row[symbol] > percentage:
                skip = 5
                orders.set_value(index, 'Order', 'enterlong')
                intrade = True
                trade = 'long'
            #if < -1%, short and hold
            if row[symbol] < -percentage:
                skip = 5
                orders.set_value(index, 'Order', 'entershort')
                intrade = True
                trade = 'short'
        if intrade:
            if skip == 0:
                if trade == 'long':
                    orders.set_value(index, 'Order', 'exitlong')
                    intrade = False
                if trade == 'short':
                    orders.set_value(index, 'Order', 'exitshort')
                    intrade = False
        skip = skip - 1
    orders = orders.dropna(how='all')
    orders.set_value(orders.index, 'Symbol', symbol)
    orders.set_value(orders.index, 'Shares', 100)
    orders.index.name = 'Date'

    long_orders = orders[(orders['Order'].str.match('enterlong'))==True]
    short_orders = orders[(orders['Order'].str.match('entershort'))==True]
    exit_orders = pd.concat([orders[(orders['Order'].str.match('exitlong'))==True],orders[(orders['Order'].str.match('exitshort'))==True]], axis=0)

    orders['Order'] = orders['Order'].replace('entershort','SELL')
    orders['Order'] = orders['Order'].replace('enterlong','BUY')
    orders['Order'] = orders['Order'].replace('exitshort','BUY')
    orders['Order'] = orders['Order'].replace('exitlong','SELL')

    orders.to_csv("./orders/orders.csv")
    return orders, long_orders, short_orders, exit_orders

if __name__=="__main__":
    symbol = 'IBM'
    # symbol = 'ML4T-220'

    # Pull the feature data for the stock, insample (2007-2009)
    y_pd, y, prices_pd = define_y(symbol)
    data_pd, data = define_features(symbol)

    y_pd = y_pd.ix[20:-5]
    y = y[:,20:-5]
    prices_pd = prices_pd.ix[20:-5]
    data_pd = data_pd.ix[20:-5]
    data = data[20:-5]

    # Pull the feature data for the stock, out (2009-2011)
    outy_pd, outy, outprices_pd = define_y(symbol, startdate_string ='12/31/09', enddate_string ='12/31/11')
    outdata_pd, outdata = define_features(symbol, startdate_string ='12/31/09', enddate_string ='12/31/11')

    outy_pd = outy_pd.ix[20:-5]
    outy = outy[:,20:-5]
    outprices_pd = outprices_pd.ix[20:-5]
    outdata_pd = outdata_pd.ix[20:-5]
    outdata = outdata[20:-5]

    #training_prices = ((y_pd+1)*prices_pd).shift(5) #unnecessary as this is equal to prices

    # compute how much of the data is training and testing
    # train_rows = int(math.floor(0.6* data.shape[0]))
    train_rows = int(math.floor(1.0* data.shape[0])) #Set to 1 so we don't xvalidate, but do in/out of sample data based on dates
    test_rows = int(math.floor(1.0* outdata.shape[0]))

    # separate out training and testing data
    trainX = data[:train_rows,0:]
    trainY = y[:,:train_rows]
    testX = outdata[:test_rows,0:]
    testY = outy[:,:test_rows]

    trainX_pd = data_pd.ix[:train_rows]
    trainY_pd = y_pd.ix[:train_rows]
    testX_pd = outdata_pd.ix[:test_rows]
    testY_pd = outy_pd[:test_rows]
    trainPrices_pd = prices_pd.ix[:train_rows]
    testPrices_pd = outprices_pd[:test_rows]

    # create a KNN learner
    knnlearner = knn.KNNLearner(k=3)
    knnlearner.addEvidence(trainX, trainY)

    # evaluate in sample
    #predY = learner.query(trainX) # get the predictions
    trainPredY = knnlearner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - trainPredY) ** 2).sum() / trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(trainPredY, y=trainY)
    print "corr: ", c[0,1]

    trainPredY_pd = pd.DataFrame(data = trainPredY, index=trainY_pd.index, columns=[symbol])
    plotTrainY =trainY_pd*trainPrices_pd + trainPrices_pd
    plotTrainPredY = trainPredY_pd*trainPrices_pd + trainPrices_pd

    trainorders, trainlong_orders, trainshort_orders, trainexit_orders = find_orders(trainPredY_pd, 0.025)
    trainportval = compute_portvals(trainorders, start_val=10000)
    print "Returns: ", trainportval.iloc[-1]/trainportval.iloc[0]

    plotTrainPredData = pd.concat([trainPrices_pd, plotTrainY, plotTrainPredY], axis=1)
    plotTrainPredData.columns = ['Training Prices', 'TrainY', 'PredY']


    #TODO Test Out of Sample
    # evaluate out of sample
    #predY = learner.query(testX) # get the predictions
    testPredY = knnlearner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - testPredY) ** 2).sum() / testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(testPredY, y=testY)
    print "corr: ", c[0,1]

    testPredY_pd = pd.DataFrame(data = testPredY, index=testY_pd.index, columns=[symbol])
    plotTestY =testPredY_pd*testPrices_pd + testPrices_pd
    plotTestPredY = testPredY_pd*testPrices_pd + testPrices_pd

    testorders, testlong_orders, testshort_orders, testexit_orders = find_orders(testPredY_pd, 0.025)
    testportval = compute_portvals(testorders, start_val=10000)
    print "Returns: ", testportval.iloc[-1]/testportval.iloc[0]

    plotTestPredData = pd.concat([testPrices_pd, plotTrainY, plotTrainPredY], axis=1)
    plotTestPredData.columns = ['Training Prices', 'TrainY', 'PredY']


    #TODO If you want to plot...
    # plot_data_2axes(plotTrainPredData, ['Predicted Y', 'Training Y'])
    plot_data(plotTrainPredData, title='Training Y/Price/Predicted Y:') #Prediction Plot
    # plot_data(plotTestPredData)
    # plot_corr(testY, testPredY)
    plot_orders(trainPrices_pd, trainlong_orders, trainshort_orders, trainexit_orders, title='In Sample Trading Data') #Entries/Exits In-Sample
    plot_orders(trainportval/trainportval[0], trainlong_orders, trainshort_orders, trainexit_orders, title='In Sample Backtest', ylabel="Cumulative Return") #In Sample Backtest
    plot_orders(testPrices_pd, testlong_orders, testshort_orders, testexit_orders, title='Out of Sample Trading Data') #Entries/Exits Out-Of-Sample
    plot_orders(testportval/testportval[0], testlong_orders, testshort_orders, testexit_orders, title='Out of Sample Backtest', ylabel="Cumulative Return") #Out of Sample Backtest


    print 'Debug Here'