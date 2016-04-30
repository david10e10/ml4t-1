"""
Test a learner.  (c) 2015 Tucker Balch
"""

import math

import matplotlib.pyplot as plt
import numpy as np

import mc3_p2.KNNLearner as knn


def func(x):
    return x*10

def threedplot(df1, df1z, df2, df2z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    ax.scatter(df1[:,0], df1[:,1], df1z, color = 'blue', marker='o')
    ax.scatter(df2[:,0], df2[:,1], df2z, color = 'red', marker='^')
    plt.show()

def plot_data(df1, df2, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    #ax = df.plot(title=title, fontsize=12, marker='o')
    plt.scatter(df1, df2, color = 'blue')
    plt.show()

if __name__=="__main__":
    inf = open('Data/ripple.csv')
    # inf = open('Data/ML4T-220.csv')
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # start_date = '2007-12-31'
    # end_date = '2009-12-31'
    # dates = pd.date_range(start_date, end_date)
    # dates = get_data(['ML4T-220'], dates).index.get_values() #fix dates so you only get when SPY is trading
    # symbols = ['ML4T-220']  # Get a LIST of symbols
    #
    # # Read in adjusted closing prices for given symbols, date range
    # prices_all = get_data(symbols, dates)  # automatically adds SPY
    # prices = prices_all[symbols]  # only portfolio symbols
    # prices = pd.concat([prices, pd.DataFrame(index=dates)], axis=1)  # all dates in prices
    # prices = prices.fillna(method='ffill')  # fillna
    # prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    #CONVERT TO NUMPY IF NEEDED
    # prices.as_matrix()

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    # trainX = data[:,0:-1]
    # trainY = data[:,-1]
    # testX = data[:,0:-1]
    # testY = data[:,-1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner.addEvidence(trainX, trainY) # train it

    # create a KNN learner
    knnlearner = knn.KNNLearner(k=3)
    knnlearner.addEvidence(trainX, trainY)

    # evaluate in sample
    #predY = learner.query(trainX) # get the predictions
    predY = knnlearner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    #predY = learner.query(testX) # get the predictions
    predY = knnlearner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    # plot_data(testY, predY)
    # threedplot(testX[0:200], testY[0:200], testX[0:200], predY[0:200])

    #learners = []
    #for i in range(0,10):
        #kwargs = {"k":i}
        #learners.append(lrl.LinRegLearner(**kwargs))
