"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # Import Orders into DataFrame
    orders = pd.read_csv(orders_file, index_col=0, parse_dates=True, sep=',')
    orders = orders.sort_index()
    start_date = orders.index[0].to_datetime()
    end_date = orders.index[-1].to_datetime()
    orders.index = orders.index
    dates = pd.date_range(start_date, end_date)
    symbols = orders.get('Symbol').unique().tolist()  # Get a LIST of symbols

    # Read in adjusted closing prices for given symbols, date range
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices = pd.concat([prices, pd.DataFrame(index=dates)], axis=1)  # all dates in prices
    prices = prices.fillna(method='ffill')  # fillna
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    #establish holder for shares
    share_transactions = pd.DataFrame(columns = symbols, index = [dates])
    share_transactions = share_transactions.fillna(value=0)

    # Establish Cash Values
    # columns = ['cash', 'buying_power'] + symbols  # Headers of DataFrame is Cash and Buying Power
    columns = ['cash'] # TODO Buying power needs to be its own DataFrame
    portvals = pd.DataFrame(columns=columns, index=[
        dates])  # initialize portfolio values, buying power = -999 until I incorporate leverage
    portvals.ix[0] = 0

    # Calculate Cash Transactions
    cash_transactions = pd.DataFrame(columns=['cash_transaction'], index=[dates])
    cash_transactions.ix[0, ['cash_transaction']] = start_val
    cash_transactions = cash_transactions.fillna(value=0)

    # Iterate through the rows in orders
    print 'Start Iteration'  # TODO
    for i in range(len(orders)):
        if orders.ix[i]['Order'] == 'BUY':
            mult = 1.0
        else:
            mult = -1.0
        share_transactions.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] = share_transactions.ix[orders.index[i].to_datetime(), orders.ix[i]['Symbol']] + mult * orders.ix[i]['Shares']
        order_cost = mult * orders.ix[i]['Shares'] * prices.ix[orders.index[i].to_datetime()][
            orders.ix[i]['Symbol']]
        # print orders.ix[i]['Symbol']
        # print orders.ix[i]['Order']
        # print order_cost
        # print orders.index[i].to_datetime()
        # print
        cash_transactions.ix[orders.index[i].to_datetime(), ['cash_transaction']] = cash_transactions.ix[orders.index[i].to_datetime(), ['cash_transaction']] - order_cost
    portvals.cash = cash_transactions.cash_transaction.cumsum() #Calculate total amt of cash in the portfolio at a given time (sum up the cash transactions)
    shares = share_transactions.cumsum() #Calculate total num of shares in the portfolio at a given time (sum up the share orders)
    portvals = pd.concat([portvals,shares * prices],axis=1)
    portfolio_value = portvals.sum(axis=1)

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_dates = dt.datetime(2008, 1, 1)
    end_dates = dt.datetime(2008, 6, 1)
#    portvals = get_data(['IBM'], pd.date_range(start_dates, end_dates))
#    portvals = portvals[['IBM']]  # remove SPY

    return portfolio_value


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_dates = dt.datetime(2008, 1, 1)
    end_dates = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

    # Compare portfolio against $SPX
    print
    print "Date Range: {} to {}".format(start_dates, end_dates)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()
