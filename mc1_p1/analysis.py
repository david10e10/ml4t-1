"""MC1-P1: Analyze a portfolio."""
#testgit

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import datetime as dt

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr, port_val = compute_portfolio_stats(prices, sv, allocs, rfr, sf)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val/port_val[0], prices_SPY/prices_SPY[0]], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, ylabel = "Normalized Price")
        pass

    return cr, adr, sddr, sr, sv

def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def compute_portfolio_stats(prices, start_val,\
    allocs=[0.1,0.2,0.3,0.4], \
    rfr = 0.0, sf = 252.0):

    normalizedPrices = prices/prices.ix[0] #Normalize the Prices
    allocatedPrices = normalizedPrices*allocs #Weight the prices according to portfolio allocation
    position_values = allocatedPrices*start_val #positions of each stock
    portfolio_value = position_values.sum(axis=1) #Combined price of each stock rolled into one
    daily_returns = ((portfolio_value/portfolio_value.shift(1)) - 1).ix[1:] #calculate Daily Return. Tomorrow - Today, remove the header (which is 0)

    cr = (portfolio_value[-1]/portfolio_value[0]) - 1
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = np.sqrt(sf)*(daily_returns-rfr).mean()/std_daily_return

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

    return cr, avg_daily_return, std_daily_return, sharpe_ratio, portfolio_value

if __name__ == "__main__":
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.2, 0.4, 0.2]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    print
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr