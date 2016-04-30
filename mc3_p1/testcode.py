
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(df1, df2, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    #ax = df.plot(title=title, fontsize=12, marker='o')
    plt.scatter(df1[:,0], df1[:,1], color = 'blue')
    plt.scatter(df2[:,0], df2[:,1], color = 'red')

    plt.show()

def threedplot(df1, df1z, df2, df2z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100
    ax.scatter(df1[:,0], df1[:,1], df1z, color = 'blue', marker='o')
    ax.scatter(df2[:,0], df2[:,1], df2z, color = 'red', marker='^')
    plt.show()

a= np.array([[1,2],[3,4]])
b=np.array([[5,6],[5,6]])
c=np.square(a-b)
d=np.sqrt(c[:,0]+c[:,1])
a3 = np.array([a, b])
a3tile = np.tile(a,(2,1,1))

inf = open('Data/ripple.csv')
data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

# compute how much of the data is training and testing
train_rows = math.floor(0.6* data.shape[0])
test_rows = data.shape[0] - train_rows

# separate out training and testing data
# trainX = data[:train_rows,0:-1]
# trainY = data[:train_rows,-1]
# testX = data[train_rows:,0:-1]
# testY = data[train_rows:,-1]

# trainX = data[:train_rows,:]
# trainY = data[:train_rows,-1]
# testX = data[train_rows:,:]
# testY = data[train_rows:,-1]

trainX = np.concatenate((data[:train_rows,:], data[:train_rows,:]), axis=1)
trainY = data[:train_rows,-1]
testX = np.concatenate((data[train_rows:,:], data[train_rows:,:]), axis=1)
testY = data[train_rows:,-1]

#TODO
# 0.trainx.shape
# 1. QueryX3d
# 2. Euclidean Distances
# 3. trainX_3d, trainY2d
# 4. orders mask

# #TODO change trainx.shape to correct variable of the training matrix
# queryX_3d = np.repeat(np.ndarray((1,1,2)),trainX.shape[0],axis=1) #initialize the 3D array of Query points
# for i in testX:
#     queryX_3d = np.concatenate((np.repeat([[i]],trainX.shape[0],axis=1), queryX_3d), axis=0) #create a 2d array of repeated pairs
# queryX_3d = np.delete(queryX_3d,0,0) #delete the initialization parameter

ndim_input = trainX.shape[1]
nrows_input = trainX.shape[0]
ndim_query = testX.shape[1]
nrows_query = testX.shape[0]

queryX_3d = np.repeat(np.ndarray((1,1,ndim_input)),nrows_input,axis=1) #initialize the 3D array of Query points
for i in testX:
    queryX_3d = np.concatenate((np.repeat(i.reshape(1,1,ndim_input),nrows_input,axis=1), queryX_3d), axis=0) #create a 2d array of repeated pairs
queryX_3d = np.delete(queryX_3d,0,0) #delete the initialization parameter

#TODO change testx.shape
# trainX_3d = np.tile(trainX, (testX.shape[0],1,1)) #Create a 3d Array of the 2d input space (X) for the trained variables
# trainY_2d = np.repeat([trainY], (testX.shape[0]), axis=0) #Create a 2d Array of the 1d output space (Y) for the trained variables. Columns = Ys, 1 row for each query pair.  As long as only have 1 output var, don't need to generalize

trainX_3d = np.tile(trainX, (nrows_query,1,1)) #Create a 3d Array of the 2d input space (X) for the trained variables
trainY_2d = np.repeat([trainY], (nrows_query), axis=0) #Create a 2d Array of the 1d output space (Y) for the trained variables. Columns = Ys, 1 row for each query pair.  As long as only have 1 output var, don't need to generalize

square_deltas = np.square(trainX_3d-queryX_3d) #Compute pairwise square distances between trained and query points
# euclidean_distances = np.sqrt(square_deltas[:,:,0] + square_deltas[:,:,1]) #compute euclidean distance for each pair

euclid_sum = np.zeros(square_deltas[:,:,0].shape)
for i in range(ndim_input):
    euclid_sum = euclid_sum + square_deltas[:,:,i]
euclidean_distances = np.sqrt(euclid_sum)

orders = np.argsort(euclidean_distances) #Find order for each position
#TODO Change 3 to K
k_positions = orders < 3 #Find min k distances
output = ((trainY_2d*(k_positions)).sum(axis=1)/3)


k=3
mask = np.zeros(orders.shape, dtype=int)
nearest_neighbors = orders[:,:k]
for i in range(nrows_query):
    row_neighbors = nearest_neighbors[i,:]
    for j in range(k):
        if (j in row_neighbors):
            mask[i,j] = 1

(mask*trainY_2d).sum(axis=1)/3

trainingDF = pd.DataFrame(trainX)

#plot_data(trainX, testX)
#threedplot(trainX, trainY, testX, testY)
threedplot(testX, testY, testX, output)

# orders = np.array([[ 0,  7,  1,  2,  8,  6,  3,  9, 11, 10,  5,  4],
#        [ 1,  7,  0,  8,  2,  6,  3,  9, 11, 10,  5,  4],
#        [ 2,  8,  7,  6,  0,  1,  3,  9, 11, 10,  5,  4],
#        [ 3,  9, 10,  5, 11,  4,  6,  8,  2,  7,  1,  0],
#        [ 4, 10,  5,  3,  9, 11,  6,  8,  2,  7,  1,  0],
#        [ 5, 10,  3,  9,  4, 11,  6,  8,  2,  7,  1,  0],
#        [ 6,  8,  2,  7,  1,  0,  3,  9, 11, 10,  5,  4],
#        [ 7,  1,  0,  8,  2,  6,  3,  9, 11, 10,  5,  4],
#        [ 8,  2,  7,  6,  1,  0,  3,  9, 11, 10,  5,  4],
#        [ 3,  9, 10,  5, 11,  4,  6,  8,  2,  7,  1,  0],
#        [10,  3,  9,  5,  4, 11,  6,  8,  2,  7,  1,  0],
#        [11,  3,  9, 10,  4,  5,  6,  8,  2,  7,  1,  0]])

# do = np.tile(orders, orders)

# orders
#
# 5.0, 2.50, 1.5
# 2.20, 1.90, 1.8
# 3.2, 2.8, 1.9
# 15.0, 10.0, 9.0
# 16.0, 10.0, 11.0
# 14.0, 12.0, 10.6


print stop