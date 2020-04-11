# k Nearest Neighbors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
from collections import defaultdict, Counter

###############
# Metrics
###############

def minkowski(x, y, p):
    """[summary]
    If p = 1, manhattan distance
    If p = 2, euclidean distance

    Arguments:
        x {[type]} -- [description]
        y {[type]} -- [description]
        p {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    if(p == "+infty"):
        return np.max(x-y)
    if(p == "-infty"):
        return np.min(x-y)
    return np.power(np.sum(np.abs(np.power(x - y, p))), 1/p)

def SquD(x, y):
    return np.sum(np.power(x - y, 2)/(x + y))

class KNeighborsClassifier():
    """

    Arguments:
        metric {}

    Attributes:
        X_train_
        y_train_
        score_
    
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
        if self.n_neighbors > X.shape[0]:
            raise ValueError("Invalid argument: k can't be larger than number of samples.")

        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        
        predictions = []
        
        # Loop through each observation
        for i in range(X.shape[0]):
            x_distances = []
            x_predictions = []

            # Loop through each training point
            for j in range(self.X_train_.shape[0]):
                # Compute and store distance between the training point and the observation
                if(self.metric == minkowski):
                    x_distances.append([self.metric(X[i], self.X_train_[j,:], self.p), j])
                else:
                    x_distances.append([self.metric(X[i], self.X_train_[j,:]), j])
            
            # Sort the list
            x_distances = sorted(x_distances)

            # Make a list of the k neighbors' labels
            for k in range(self.n_neighbors):
                index = x_distances[k][1]
                x_predictions.append(self.y_train_[index])
            
            # Save the most common label
            predictions.append(Counter(x_predictions).most_common(1)[0][0])

        return predictions

    def score(self, guessed_labels, true_labels):
        """Computes the accuracy of the random forest.
        
        Arguments:
            guessed_labels {list} -- the list of labels guessed by the decision tree
            true_labels {list} -- the list of the real labels
        
        Returns:
            float -- the accuracy
        """
        score = 0
        n = len(guessed_labels)
        for i in range(n):
            score += (int(guessed_labels[i]) == int(true_labels[i]))
        score = score/n
        return score

    def findFeatureImportance(self, X, y):
        """Calculate feature importances according to score function of OurDecisionTreeClassifier.

        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the training dataset without labels
            y {numpy-ndarray} of shape (n_samples,) -- the labels of the training dataset

        Returns:
            tuple of list -- the features associated with their importance
        """

        n_features = X.shape[1]
        feat_imp = {}

        # Make predictions
        predictions_without_shuffle = self.predict(X)

        # Compute the error value with all columns in place without shuffling
        base = self.score(predictions_without_shuffle, y)

        # Loop through each column 
        for ind in range(n_features):
            print("ind = ", ind)
            X_new = X.copy()

            # Shuffle the column
            np.random.shuffle(X_new[:, ind])

            # Make predictions again
            predictions_with_shuffle = self.predict(X_new)

            # Compute change in error term as compared to predictions_without_shuffle
            # Greater change means more importance 
            feat_imp[ind] = abs(self.score(predictions_with_shuffle, y) - base)

            del X_new

        features, importances = zip(*feat_imp.items())
        return features, importances

