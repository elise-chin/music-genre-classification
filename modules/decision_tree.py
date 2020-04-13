# Decision Tree

"""
Ce module contient :
    - la classe `Node` pour representer un noeud d'un arbre de decision
    - la classe `OurDecisionTreeClassifier` 
    - les methodes gini, infoGain, generateCut, bestCut, partition et lenDictValues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
from collections import defaultdict, Counter

#######################
#        Node
#######################

class Node:
    """Represents a node of a decision tree.

    Argument:
        data {dict} -- the key represents a label, the values a list of data each represented by a vector
        
    Attributes:
        father {Node} -- the parent node
        left_son {Node} -- the left child node
        right_son {Node} -- the right child node
        question {tuple} -- (chosenFeature, decisional_value) which records the best question to ask at this node of the tree. The question is "chosenFeature <= decisional_value ?"
    """
    def __init__(self, data):
        """Node instanciation. 
        
        Arguments:
            data {dict} -- the key represents a label, the values a list of data each represented by a vector
        """
        self.father = None
        self.left_son = None
        self.right_son = None
        self.data = data
        self.question = None
    
    def displayTree(self, depth):
        """Prints each node's condition of the binary tree. A leaf is represented by None.
        
        Arguments:
            depth {int} -- the tree's depth
        """
        for i in range(depth):
            if(i < depth - 1):
                print("   ", end = " ")
            if(i == depth - 1):
                print("|-->", end = " ")
        if(self.question == None):
            print("None")
        else:
            print("({0[0]}, {0[1]:.2f})".format(self.question))
        if(self.left_son != None):
            self.left_son.displayTree(depth + 1)
        if(self.right_son != None):
            self.right_son.displayTree(depth + 1)
        
    def depth(self):
        """Computes the tree's depth.
        
        Returns:
            int -- the tree's depth
        """
        if self.right_son == None and self.left_son == None:
            return 0
        else:
            max_left, max_right = 0, 0
            if(self.left_son != None):
                depth_left = self.left_son.depth()
                if depth_left > max_left:
                    max_left = depth_left
            if(self.right_son != None):
                depth_right = self.right_son.depth()
                if depth_right > max_right:
                    max_right = depth_right
            m = max_left if max_left >= max_right else max_right
            return m + 1


##############################
#       Decision Tree
##############################

class OurDecisionTreeClassifier():
    """A decision tree classifier.
    
    Arguments:
        n_cuts {int} -- the number of cuts tested to find the best cut
        max_depth {int} -- the maximal depth of a decision tree
        max_features {int, "sqrt"} -- the number of features when `fit` is performed

    Attributes:
        tree_ {Node} -- the root of the decision tree
        n_features_ {int} -- the total number of features in the training set
        features_ {list} -- the list of features randomly chosen when `fit` is performed
        score_ {float} -- the test set accuracy
    """
    def __init__(self, n_cuts = 40, max_depth = 20, max_features = "sqrt"):
        self.n_cuts = n_cuts
        self.max_depth = max_depth
        self.max_features = max_features

    def get_params(self):
        """Get the parameters of the decision tree.
        
        Returns:
            dict -- estimator parameters
        """
        params = {
            'n_cuts': self.n_cuts,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
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
        """Builds the decision tree from the training set (X, y).
        
        Arguments:
            X {numpy_ndarray} of shape (n_samples, n_features) -- the training input samples without labels
            y {numpy_ndarray} of shape (n_samples,) -- the classes labels
        
        Returns:
            OurDecisionTreeClassifier -- self
        """
        def decisionTree(node, n_cuts, features, max_depth):
            """Builds the decision tree.
            
            Arguments:
                node {Node} -- the root node
                n_cuts {int} -- the number of cuts tested to find the best cut
                features {list} -- the list of features
                max_depth {int} -- the maximal depth of a decision tree
            
            Returns:
                Node -- the root of the decision tree
            """
            gain, c1, c2, question = bestCut(node, n_cuts, features)
            
            # There is no further information gain.
            if(gini(node.data) == 0 or gain == 0 or node.depth() == max_depth):
                return node # Leaf
            
            # The best cut has been found. We can then partition the dataset.
            partition(node, c1, c2, question)
            
            # Recursively builds the tree.
            decisionTree(node.left_son, n_cuts, features, max_depth-1)
            decisionTree(node.right_son, n_cuts, features, max_depth-1)
            return node # Decision node

        self.n_features_ = X.shape[1]

        if(self.max_features == "sqrt"):
            self.max_features = int(sqrt(self.n_features_) + 1)
        elif(not isinstance(self.max_features, int) or self.max_features > self.n_features_):
            raise ValueError("Invalid parameter: max_features should be an integer <= n_features or \"sqrt\"")
        
        # Random sampling of the features
        self.features_ = rd.sample([i for i in range(self.n_features_)], self.max_features)
        
        # Transform the training set into a dictionary
        X_dict = defaultdict(list)
        for i in range(X.shape[0]):
            X_dict[y[i]].append(X[i,:])

        # Build the tree
        self.tree_ = decisionTree(node=Node(X_dict), n_cuts=self.n_cuts, features=self.features_, max_depth=self.max_depth)

        return self

    def predict(self, X):
        """Predict class for X.
    
        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the test input to classify
        
        Returns:
            list -- the predicted classes
        """
        predictions = []
        for i in range(X.shape[0]):
            node = self.tree_
            exists_son = True
            while(node != None and node.question != None and exists_son):
                if(X[i,:][node.question[0]] <= node.question[1]):
                    node = node.left_son
                else:
                    node = node.right_son
                if(node.left_son == None and node.right_son == None):
                    exists_son = False
            majoritary_index = np.argmax([len(points) for points in list(node.data.values())])
            key = list(node.data.keys())[majoritary_index]
            predictions.append(key)
        return predictions


##################
# Methodes utiles
##################

def gini(data):
    """Gini Impurity for a set of data saved in a dictionary.
    The Gini Impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
    
    Arguments:
        data {dict} -- the dataset
    
    Returns:
        float -- gini impurity
    """
    probs = [len(L) for L in data.values()] # number of data per label
    probs = [prob/sum(probs) for prob in probs] 
    return 1 - sum([i ** 2 for i in probs])

def infoGain(c, c1, c2):
    """Computes the information gain from a cut. It is used to decide which feature to split on at each step in building the decision tree.
    
    Arguments:
        c {dict} -- the initial dataset
        c1 {dict} -- the first part of the dataset formed after the cut
        c2 {dict} -- the second part of the dataset formed after the cut
    
    Returns:
        float -- information gain after the split
    """
    p = float(lenDictValues(c1)/lenDictValues(c))
    return gini(c) - (p*gini(c1) + (1-p)*gini(c2))

def generateCut(node, features):
    """Decides randomly which feature to split on and generate the two datasets stem from the cut.
    
    Arguments:
        node {Node} -- the node associated to the dataset we want to split
        features {list} -- the list of features
    
    Returns:
        (Node, Node, one-element list, float) -- the two datasets, the chosen feature and the decisional value which compose the question.
    """
    # Feature random choice.
    random_feature = rd.sample(features, 1)
    
    points = []
    values = node.data.values()
    c1, c2 = defaultdict(list), defaultdict(list)
    decisional_value = 0
    
    if len(values) != 0:
        # Put all the values in the list points ...
        for line in node.data.values():
            points.extend(line)
        
        # ... in order to randomly choose the decisional value
        decisional_value = float(rd.choice([v[random_feature] for v in points]))
        
        # Construct the two datasets stem from the cut.
        for key, l in node.data.items():
            for v in l:
                if v[random_feature] < decisional_value:
                    c1[key].append(v)
                else:
                    c2[key].append(v)
                    
    return c1, c2, random_feature, decisional_value

def bestCut(node, nb_cut, features):
    """Finds the best cut between nb_cut possible cut
    
    Arguments:
        node {Node} -- the node associated to the dataset we want to split
        nb_cut {int} -- the number of cuts tested
        features {list} -- the list of features
    
    Returns:
        (float, dict, dict, int, float) -- the best information gain, the two datasets stem from the best cut, the feature and decisional value associated
    """
    best_c1, best_c2, best_feature, best_decisional_value = generateCut(node, features) # to keep track of the split
    best_info = infoGain(node.data, best_c1, best_c2) # to keep track of the best information gain.
    
    for _ in range(nb_cut-1):
        c1, c2, random_feature, decisional_value = generateCut(node, features)
        
        # Skip the cut if the dataset was not divided.
        if len(c1) == 0 or len(c2) == 0:
            continue
        
        info = infoGain(node.data, c1, c2)
        if(info > best_info):
            best_info = info
            best_c1, best_c2, best_feature, best_decisional_value = c1, c2, random_feature, decisional_value
            
    return best_info, best_c1, best_c2, (best_feature[0], best_decisional_value)

def partition(node, c1, c2, question):
    """Partitions the current dataset saved in the node in two datasets c1 and c2 and updates the question of the node.
    
    Arguments:
        node {Node} -- the node associated to the dataset we want to split
        c1 {dict} -- the first part of the dataset formed after the cut
        c2 {dict} -- the second part of the dataset formed after the cut
        question {tuple of int} -- the chosen feature and decisional value
    """
    node.question = question
    node.left_son = Node(c1)
    node.left_son.father = node
    node.right_son = Node(c2)
    node.right_son.father = node


########################
# Pour les dictionnaires
########################

def lenDictValues(dic):
    """Computes the total number of data in a dataset represented by a dictionary. Each value is a list.
    
    Arguments:
        dic {dict} -- the dataset
    
    Returns:
        int -- the number of data
    """
    s = [len(l) for l in dic.values()]
    return sum(s)