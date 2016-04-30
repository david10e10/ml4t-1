"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np


class KNNLearner(object):
    def __init__(self, k, verbose=False):
        self.k = k
        pass  # move along, these aren't the drones you're looking for

    def addEvidence(self, Xtrain, Ytrain):
        """
        Adds raw data to learn.  Dataset is in format of X1, X2, Y

        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # build and save the datastructure
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def euclid_distance_2d(self, pq):
        """

        :param p1:
        :param p2:
        :param q:
        :return:
        """
        p1 = pq[0]
        p2 = pq[1]
        q1 = pq[2]
        q2 = pq[3]

        d = np.power([np.power(p1 - q1, 2) + np.power(p2 - q2, 2)], 0.5)

        return d

    def euclid_distance_array(self, p):
        """
        Calculates euclidean distance for a pair of 2dimensional input points <p1, p2> to <q1, q2>
        :param p1: 1st dimension of 1st point
        :param p2: 2nd dimension of 1st point
        :param q1: 1st dimension of 2nd point
        :param q2: 2nd dimension of 2nd point
        :return: euclidean distance metric to the pair of input values
        """
        p1 = np.ones([self.Xtrain.shape[0], 1]) * p[0]
        p2 = np.ones([self.Xtrain.shape[0], 1]) * p[1]

        d_array = np.apply_along_axis(self.euclid_distance_2d, 1, np.concatenate((p1, p2, self.Xtrain), 1))

        return d_array

    def query_original(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # calculate distances for nearest k neighbors
        # dtest = np.apply_along_axis(self.euclid_distance_array, 1, Xtest)
        distances = np.ndarray([1, len(self.Xtrain)])
        orders = np.ndarray([1, len(self.Xtrain)])
        mindistances = np.ndarray([1, 3])
        avgY = np.ndarray([1, 1])
        for i in Xtest:
            tempdist = self.euclid_distance_array(i).transpose()
            temporder = np.argsort(tempdist)
            tempmindist = tempdist[0, [temporder[0, 0:self.k]]]
            tempavgY = self.Ytrain[[temporder[0, 0:self.k]]].mean()
            distances = np.concatenate([distances, tempdist], 0)
            orders = np.concatenate([orders, temporder], 0)
            mindistances = np.concatenate([mindistances, tempmindist], 0)
            avgY = np.insert(avgY, -1, tempavgY)
        distances = np.delete(distances, 0, 0)
        orders = np.delete(orders, 0, 0)
        orders = orders.astype(int)
        mindistances = np.delete(mindistances, 0, 0)
        avgY = np.delete(avgY, 0, 0)

        # distances[0,[orders[0][0:3]]]
        # identify y vals for nearest neighbors

        # return average for k-yvals
        return avgY

    def query_old2(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        queryX_3d = np.repeat(np.ndarray((1, 1, 2)), self.Xtrain.shape[0], axis=1)  # initialize the 3D array of Query points
        for i in Xtest:
            queryX_3d = np.concatenate((queryX_3d, np.repeat([[i]], self.Xtrain.shape[0], axis=1)), axis=0)  # create a 2d array of repeated pairs
            #queryX_3d = np.concatenate((np.repeat([[i]], self.Xtrain.shape[0], axis=1), queryX_3d), axis=0)  # create a 2d array of repeated pairs
        queryX_3d = np.delete(queryX_3d, 0, 0)  # delete the initialization parameter

        trainX_3d = np.tile(self.Xtrain, (Xtest.shape[0], 1, 1))  # Create a 3d Array of the 2d input space (X) for the trained variables
        trainY_2d = np.repeat([self.Ytrain], (Xtest.shape[0]), axis=0)  # Create a 2d Array of the 1d output space (Y) for the trained variables. Columns = Ys, 1 row for each query pair

        square_deltas = np.square(trainX_3d - queryX_3d)  # Compute pairwise square distances between trained and query points
        euclidean_distances = np.sqrt(square_deltas[:, :, 0] + square_deltas[:, :, 1])  # compute euclidean distance for each pair
        orders = np.argsort(euclidean_distances)  # Find order for each position
        #kclosestorders = euclidean_distances[:]
        #k_positions = orders[0:self.k] # Find min k distances
        # dim1array = np.array([],dtype=int)
        # for i in range(orders.shape[0]):
        #     dim1array = np.concatenate((dim1array,i*np.ones(3, dtype=int)), axis = 0)
        # dim2array = np.reshape(orders[:,0:self.k],[1,orders.shape[0]*self.k])
        out = euclidean_distances[dim1array, dim2array]

        return out

    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        ndim_input = self.Xtrain.shape[1]
        nrows_input = self.Xtrain.shape[0]
        ndim_query = Xtest.shape[1]
        nrows_query = Xtest.shape[0]

        queryX_3d = np.repeat(np.ndarray((1,1,ndim_input)),nrows_input,axis=1) #initialize the 3D array of Query points
        for i in Xtest:
            queryX_3d = np.concatenate((queryX_3d, np.repeat(i.reshape(1,1,ndim_input),nrows_input,axis=1)), axis=0) #create a 2d array of repeated pairs
        queryX_3d = np.delete(queryX_3d,0,0) #delete the initialization parameter

        trainX_3d = np.tile(self.Xtrain, (nrows_query,1,1)) #Create a 3d Array of the 2d input space (X) for the trained variables
        trainY_2d = np.repeat([self.Ytrain], (nrows_query), axis=0) #Create a 2d Array of the 1d output space (Y) for the trained variables. Columns = Ys, 1 row for each query pair.  As long as only have 1 output var, don't need to generalize

        square_deltas = np.square(trainX_3d - queryX_3d)  # Compute pairwise square distances between trained and query points

        # euclid_sum = np.zeros(square_deltas[:,:,0].shape)
        # for i in range(ndim_input):
        #     euclid_sum = euclid_sum + square_deltas[:,:,i]
        euclidean_distances = np.sqrt(square_deltas.sum(axis=2)) #TODO Inspect/analyze this axis

        orders = np.argsort(euclidean_distances)  # Find order for each position

        mask = np.zeros(orders.shape, dtype=int)
        nearest_neighbors = orders[:,:self.k]
        for i in range(nrows_query):
            row_neighbors = nearest_neighbors[i,:]
            for j in range(nrows_input):
                if (j in row_neighbors):
                    mask[i,j] = 1

        out = (mask*trainY_2d).sum(axis=1)/3

        return out

if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
