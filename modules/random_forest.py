# Random Forest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
from collections import defaultdict, Counter

from modules.decision_tree import OurDecisionTreeClassifier

class OurRandomForestClassifier():
    """A random forest classifier.
    
    Arguments:
        n_trees {int} -- the number of trees in the forest
        n_samples {int} -- the number of data points placed in a node before the node is split
        n_cuts {int} -- the number of cuts tested to find the best cut
        max_depth {int} -- the maximal depth of each tree
        max_features {int, "sqrt"} -- the number of features when `fit` is performed
    
    Attributes:
        base_estimator_ {OurDecisionTreeClassifier} -- the class used to create the list of trees
        trees_ {list of OurDecisionTreeClassifier} -- the collection of fitted sub-estimators
        n_classes_ {int} -- the number of classes of the training set
        score_ {float} -- the test set accuracy
    """

    def __init__(self, n_trees = 100, n_samples = 100, n_cuts = 40, max_depth = 45, max_features = 30):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.n_cuts = n_cuts
        self.max_depth = max_depth
        self.max_features = max_features
        self.base_estimator_ = OurDecisionTreeClassifier(n_cuts, max_depth, max_features)

    def get_params(self):
        """Get the parameters of the random forest.
        
        Returns:
            dict -- the parameters
        """
        params = {
            'n_trees': self.n_trees,
            'n_samples': self.n_samples,
            'n_cuts': self.n_cuts,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
        }
        return params

    def set_params(self, params):
        """Set the parameters of the random forest.
        
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
        """Build a forest of trees from the training set (X, y).
        
        Arguments:
            X {numpy_ndarray} of shape (n_samples, n_features) -- the training input samples without labels
            y {numpy_ndarray} of shape (n_samples,) -- the classes labels
        
        Returns:
            OurRandomForestClassifier -- self
        """

        self.n_classes_ = np.unique(y)
        self.trees_ = []

        # Loop through each tree
        for _ in range(self.n_trees):
            
            # Instanciation of the decision tree
            if(self.max_features == "sqrt"):
                self.max_features = int(sqrt(X.shape[1]) + 1)
            elif(not isinstance(self.max_features, int) or self.max_features > X.shape[1]):
                raise ValueError("Invalid parameter: max_features should be an integer <= number of features of the training set or \"sqrt\"")
            
            dt = OurDecisionTreeClassifier(self.n_cuts, self.max_depth, self.max_features)

            # Random sampling of size self.n_samples in X
            if self.n_samples > X.shape[0]:
                raise ValueError("Invalid argument: n_samples should be <= X.shape[0]")
            indexes = rd.sample([i for i in range(X.shape[0])], self.n_samples)
            X_sample = [X[i,:] for i in indexes]
            y_sample = [y[i] for i in indexes]
            
            # Build the tree
            dt.fit(np.array(X_sample), np.array(y_sample))
            self.trees_.append(dt)

        return self

    def predict(self, X):
        """Predict class for X.
        
        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the input to classify
        
        Returns:
            list of int -- the predicted classes
        """
        all_predictions = np.array([tree.predict(X) for tree in self.trees_])
        predictions = []
        for i in range(X.shape[0]):
            predictions.extend([Counter(all_predictions[:,i]).most_common(1)[0][0]])
        return predictions

