# K Nearest Neighbors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
from collections import defaultdict, Counter

def minkowski(x, y, p):
    """Minkowski distance.

    Arguments:
        x {list or numpy-ndarray} -- first observation
        y {list or numpy-ndarray} -- second observation
        p {int} -- power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    
    Returns:
        float -- Minkowski distance of power p between x and y
    """
    if(p == "+infty"):
        return np.max(x-y)
    if(p == "-infty"):
        return np.min(x-y)
    return np.power(np.sum(np.abs(np.power(x - y, p))), 1/p)

class OurKNeighborsClassifier():
    """Classifier implementing the k-nearest neighbors vote.

    Arguments:
        n_neighbors {int} -- number of neighbors to find for a point
        metric {callable} -- the distance metric to use
        p {int} -- power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    Attributes:
        X_train_ {numpy-ndarray} of shape (n_samples, n_features) -- the training input 
        y_train_ {numpy-ndarray} of shape (n_samples, ) -- the classes labels
        score_ {float} -- the accuracy score of the model
    
    """
    def __init__(self, n_neighbors=1, metric=minkowski, p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def get_params(self):
        """Get the parameters of the k nearest neighbors classifier.
        
        Returns:
            dict -- estimator parameters
        """
        params = {
            'n_neighbors': self.n_neighbors,
            'metric': self.metric,
            'p': self.p
        }
        return params

    def set_params(self, params):
        """Set the parameters of the decision tree.
        
        Arguments:
            params {dict} -- estimator parameters
        
        Returns:
            self -- estimator instance
        """
        valid_params = self.get_params().keys()
        for key in params.keys():
            if key not in valid_params:
                raise ValueError("Invalid parameter %s for estimator %s."%(key, self.__class__))
            setattr(self, key, params[key])
        return self

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.
        
        Arguments:
            X {numpy_ndarray} of shape (n_samples, n_features) -- the training input samples without labels
            y {numpy_ndarray} of shape (n_samples,) -- the classes labels
        
        Returns:
            OurKNeighborsClassifier -- self
        """
        if self.n_neighbors > X.shape[0]:
            raise ValueError("Invalid argument: k can't be larger than number of samples.")

        self.X_train_ = X
        self.y_train_ = y
        return self

    def kneighbors(self, x, n_neighbors):
        """Finds the k-neighbors of a point.
        
        Arguments:
            x {list or numpy-ndarray} of shape (n_samples, ) -- a point
            n_neighbors {int} -- number of neighbors to find for x
        
        Returns:
            list of float, list of int -- the n_neighbors nearest neighbors distances from x and their indexes
        """
        neigh_distances = []
        neigh_indexes = []

        # Loop through each training point
        for i in range(self.X_train_.shape[0]):

            # Compute and store distance between the training point and the observation
            if(self.metric == minkowski):
                neigh_distances.append([self.metric(x, self.X_train_[i,:], self.p), i])
            else:
                neigh_distances.append([self.metric(x, self.X_train_[i,:]), i])
        
        # Sort the list
        neigh_distances = sorted(neigh_distances)

        # Make a list of the k neighbors' indexes
        for k in range(self.n_neighbors):
            index = neigh_distances[k][1]
            neigh_indexes.append(index)
        
        return neigh_distances, neigh_indexes


    def predict(self, X):
        """Predict class for X.
    
        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the test input to classify
        
        Returns:
            list -- the predicted classes
        """
        predictions = []
        
        # Loop through each observation
        for i in range(X.shape[0]):
            x_predictions = []

            # Get the k nearest neighbors indexes of each observation
            _, neigh_indexes = self.kneighbors(X[i,:], self.n_neighbors)
            
            # Make a list of the k neighbors' labels
            x_predictions = [self.y_train_[index] for index in neigh_indexes]

            # Save the most common label
            predictions.append(Counter(x_predictions).most_common(1)[0][0])

        return predictions