import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
import marketsim as ms

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def plot_data(df, long_orders, short_orders, exit_orders, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ymin = df['Lower Band'].min() - 1
    ymax = df['Upper Band'].max() + 1

    plt.vlines(long_orders.index, color='green',ymin=ymin, ymax=ymax)
    plt.vlines(short_orders.index, color='red',ymin=ymin, ymax=ymax)
    plt.vlines(exit_orders.index, color='black',ymin=ymin, ymax=ymax)
    plt.show()

def define_bollingerband(symbol):
    # Import Orders into DataFrame (CURRENTLY HAS ALL DATES including non-trading)
    start_date = pd.to_datetime('12/31/09') #StartDate per Instructions
    end_date = pd.to_datetime('12/31/11') #EndDate per Instructions
    dates = pd.date_range(start_date, end_date)

    symbols = [symbol, '$SPX']

    # Read in adjusted closing prices for given symbols, date range
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[[symbol]]  # only portfolio symbols

    # Compute SMA
    sma = pd.rolling_mean(prices, 20)
    sma.columns = ['SMA']

    # Compute Std Dev
    std_dev = pd.rolling_std(prices, 20)
    std_dev.columns = ['Standard Deviation']

    # Calculate Bollinger Band Limits
    lower_bband = sma.subtract(2*std_dev.ix[:,0], axis=0)
    lower_bband.columns = ['Lower Band']
    upper_bband = sma.add(2*std_dev.ix[:,0], axis=0)
    upper_bband.columns = ['Upper Band']

    # Combine All Data into 1 dataframe
    data = pd.concat([prices, sma, lower_bband, upper_bband], axis = 1)

    # Compute 4 Statuses
    below_lower = pd.DataFrame(data['IBM']<data['Lower Band'], columns = ['Below Lower']) #Low Points: Identify where Stock < Lower Band
    above_sma = pd.DataFrame(data['IBM']>data['SMA'], columns = ['Above SMA']) #Mid Points: Identify where Stock > SMA
    above_upper = pd.DataFrame(data['IBM']>data['Upper Band'], columns = ['Above Upper']) #High Points: Identify where Stock > Upper Band
    status = pd.concat([below_lower, above_sma, above_upper], axis = 1)
    status_shift = status.shift(1) #aka 'Yesterday'

    # Compute 4 Actions (get lazy and do iterator)
    position_action = pd.DataFrame(index=prices.index, columns = ['Order']) #initialize the Orders Dataframe
            #data['IBM']-data['IBM'].shift(1) #n compared to n-1
    position_action[(status_shift['Below Lower']==True)&(status['Below Lower']==False)]='enterlong'  #Enter Long: Yesterday Below Lower -> Today Above Lower
    position_action[(status_shift['Above SMA']==False)&(status['Above SMA']==True)]='exitlong'       #Exit Long: Yesterday Below SMA -> Today Above SMA
    position_action[(status_shift['Above Upper']==True)&(status['Above Upper']==False)]='entershort' #Enter Short: Yesterday Above Upper -> Today Below Upper
    position_action[(status_shift['Above SMA']==True)&(status['Above SMA']==False)]='exitshort'      #Exit Short: Yesterday Above SMA -> Today Below SMA
    position_action = position_action.dropna()

    entered_posn = False
    drops = pd.DataFrame(index=position_action.index, columns = ['change']) #initialize the Orders Dataframe
    for index, row in position_action.iterrows():
        print index
        print row
        if entered_posn == False:
            #calculate enters
            if (row[0] == 'enterlong') or (row[0] == 'entershort'):
                entered_posn = True
            else: #exitlong or exitshort
                #position_action.drop(index)
                drops.loc[index] = 1
        else:
            if (row[0] == 'exitlong') or (row[0] == 'exitshort'):
                entered_posn = False
            else: #enterlong or entershort
                #position_action.drop(index)
                drops.loc[index] = 1
    drops = drops.fillna(0)
    position_action = position_action[drops['change']==0]

    orders = pd.DataFrame(index=position_action.index, columns = [['Symbol', 'Order', 'Shares']])
    orders.index.name = 'Date'
    orders['Symbol'] = 'IBM'
    orders['Shares'] = 100
    orders['Order'] = orders['Order'].fillna(position_action['Order'])
    orders['Order'] = orders['Order'].replace('entershort','SELL')
    orders['Order'] = orders['Order'].replace('enterlong','BUY')
    orders['Order'] = orders['Order'].replace('exitshort','BUY')
    orders['Order'] = orders['Order'].replace('exitlong','SELL')

    long_orders = position_action[(position_action['Order'].str.match('enterlong'))==True]
    short_orders = position_action[(position_action['Order'].str.match('entershort'))==True]
    exit_orders = pd.concat([position_action[(position_action['Order'].str.match('exitlong'))==True],position_action[(position_action['Order'].str.match('exitshort'))==True]], axis=0)

    orders.to_csv("./orders/orders.csv")

    #data = pd.concat([data, position_action], axis=1)

    # Plot the Data
    plot_data(data, long_orders, short_orders, exit_orders)

    return

def test_code():
    define_bollingerband('IBM')

if __name__ == "__main__":
    test_code()
    print "Hello"