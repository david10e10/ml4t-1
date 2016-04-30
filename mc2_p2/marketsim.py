"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    # this is the function the autograder will call to test your code

    # Import Orders into DataFrame
    orders = pd.read_csv(orders_file, index_col=0, parse_dates=True, sep=',')
    orders = orders.sort_index()
    start_date = orders.index[0].to_datetime()
    end_date = orders.index[-1].to_datetime()
    orders.index = orders.index
    dates = pd.date_range(start_date, end_date)
    dates = get_data(['$SPX'], dates).index.get_values() #fix dates so you only get when SPY is trading
    symbols = orders.get('Symbol').unique().tolist()  # Get a LIST of symbols

    # Read in adjusted closing prices for given symbols, date range
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices = pd.concat([prices, pd.DataFrame(index=dates)], axis=1)  # all dates in prices
    prices = prices.fillna(method='ffill')  # fillna
    prices_SPY = prices_all['$SPX']  # only SPY, for comparison later

    #establish holder for shares
    share_transactions = pd.DataFrame(columns = symbols, index = [dates])
    share_transactions = share_transactions.fillna(value=0)

    # Establish Cash Values
    # columns = ['cash', 'buying_power'] + symbols  # Headers of DataFrame is Cash and Buying Power
    columns = ['cash'] # TODO Buying power needs to be its own DataFrame
    positions = pd.DataFrame(columns=columns, index=[
        dates])  # initialize portfolio values, buying power = -999 until I incorporate leverage
    positions.ix[0] = 0

    # Calculate Cash Transactions
    cash_transactions = pd.DataFrame(columns=['cash_transaction'], index=[dates])
    cash_transactions.ix[0, ['cash_transaction']] = start_val
    cash_transactions = cash_transactions.fillna(value=0)

    # Calculate Leverage
    leverage =  pd.DataFrame(columns = ['leverage'], index=[dates])
    leverage.ix[:,['leverage']] = 0
    #positions.ix[:,symbols].abs().sum(axis=1)

    # Iterate through the rows in orders
    for i in range(len(orders)):
        if orders.ix[i]['Order'] == 'BUY':
            mult = 1.0
        else:
            mult = -1.0
        share_transactions.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] = share_transactions.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] + mult * orders.ix[i]['Shares']
        order_cost = mult * orders.ix[i]['Shares'] * prices.ix[orders.index[i].to_datetime()][
            orders.ix[i]['Symbol']]
        print orders.ix[i]['Symbol']
        print orders.ix[i]['Order']
        print order_cost
        print orders.index[i].to_datetime()
        print
        cash_transactions.ix[orders.index[i].to_datetime(), ['cash_transaction']] = cash_transactions.ix[orders.index[i].to_datetime(), ['cash_transaction']] - order_cost
    positions.cash = cash_transactions.cash_transaction.cumsum() #Calculate total amt of cash in the portfolio at a given time (sum up the cash transactions)
    shares = share_transactions.cumsum() #Calculate total num of shares in the portfolio at a given time (sum up the share orders)
    positions = pd.concat([positions,shares * prices],axis=1)
    portfolio_value = positions.sum(axis=1)

    #The trick to getting leverage in there is to concatenate the length of the positions dataframe every time an order is executed.  That seems like a lot of work that I don't want to do right now.
    # I'll take the 4% hit

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_dates = dt.datetime(2008, 1, 1)
    end_dates = dt.datetime(2008, 6, 1)
#    positions = get_data(['IBM'], pd.date_range(start_dates, end_dates))
#    positions = positions[['IBM']]  # remove SPY

    return portfolio_value

def compute_portfolio_stats(portfolio_value, \
    rfr = 0.0, sf = 252.0):

    daily_returns = ((portfolio_value/portfolio_value.shift(1)) - 1).ix[1:] #calculate Daily Return. Tomorrow - Today, remove the header (which is 0)

    cr = (portfolio_value.ix[-1]/portfolio_value.ix[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = np.sqrt(sf)*(daily_returns-rfr).mean()/std_daily_return

    start_date = portfolio_value.index[0].to_datetime()
    end_date = portfolio_value.index[-1].to_datetime()

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

    return cr, avg_daily_return, std_daily_return, sharpe_ratio, portfolio_value, start_date, end_date

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    sv = 10000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    #start_dates = dt.datetime(2008, 1, 1)
    #end_dates = dt.datetime(2008, 6, 1)
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    #cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, portfolio_value, start_dates, end_dates = compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY, spy_value, start_dates, end_dates  = compute_portfolio_stats(get_data(['$SPX'], dates = pd.date_range(start_dates, end_dates)))
    portfolio_value = portfolio_value.to_frame()
    portfolio_value.columns=[['Portfolio']]
    vals = pd.concat([portfolio_value/portfolio_value.iloc[0], spy_value/spy_value.iloc[0]], axis = 1)
    plot_data(vals)

    # Compare portfolio against $SPX
    print
    print "Date Range: {} to {}".format(start_dates, end_dates)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()
