"""
Ce module contient :
    - la classe `Node` pour representer un noeud d'un arbre de decision
    - la classe `OurDecisionTreeClassifier` 
    - la classe `OurRandomForestClassifier`
    - d'autres methodes tels que gini, infoGain...
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin

#######################
#        Node
#######################

class Node:
    """Represents a node of a decision tree.

    Attributes:
        father {Node} -- the parent node
        left_son {Node} -- the left child node
        right_son {Node} -- the right child node
        data {dict} -- the key represents a label, the values a list of data each represented by a vector
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


###################################################
#           OurDecisionTreeClassifier
###################################################

class OurDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """A decision tree classifier.
    
    Arguments:
        n_cuts {int} -- the number of cuts tested to find the best cut
        max_depth {int} -- the maximal depth of a decision tree

    Attributes:
        tree_ {Node} -- the root of the decision tree
        n_samples_ {int} --
        n_features_ {int} -- the number of features when `fit` is performed
        features_ {list} -- the list of features (je sais pas encore si c nessaire)
    """
    def __init__(self, n_cuts = 10, max_depth = 10):
        self.n_cuts = n_cuts
        self.max_depth = max_depth

    def fit(self, X, y, features):
        """Build a decision tree classifier from the training set X.
        
        Arguments:
            X {numpy_ndarray} -- The training input samples without labels
            y {numpy_ndarray} -- The classes labels
            features {list} -- The list of features

        Returns:
            OurDecisionTreeClassifier -- self, the fitted estimator
        """
        def _build(node, features, depth_max):
            gain, c1, c2, question = bestCut(node, self.n_cuts, features)
        
            # There is no further information gain.
            if(gini(node.data) == 0 or gain == 0 or node.depth() == depth_max):
                return node # Leaf
            
            # The best cut has been found. We can then partition the dataset.
            partition(node, c1, c2, question)
            
            # Recursively builds the tree.
            _build(node.left_son, features, depth_max-1)
            _build(node.right_son, features, depth_max-1)

        self.n_samples_ = X.shape[0]
        self.n_features_ = len(features)
        self.features_ = features
        
        # Associate each data point in X to its label in y 
        X_dict = defaultdict(list)
        for k in range(X.shape[0]):
            X_dict[y[k]].append(X[k])

        # Build the tree
        root = Node(X_dict)
        _build(node=root, features=features, depth_max=self.max_depth)
        self.tree_ = root

        return self

    def predict(self, X):
        """Predict class for X.
    
        Arguments:
            X {numpy-ndarray} -- the input to classify
        
        Returns:
            int -- the predicted class
        """
        exists_son = True
        while(self.tree_ != None and self.tree_.question != None and exists_son):
            if(X[self.tree_.question[0]] <= self.tree_.question[1]):
                self.tree_ = self.tree_.left_son
            else:
                self.tree_ = self.tree_.right_son
            if(self.tree_.left_son == None and self.tree_.right_son == None):
                exists_son = False
        majoritary_index = np.argmax([len(points) for points in list(self.tree_.data.values())])
        key = list(self.tree_.data.keys())[majoritary_index]
        return int(key)

#####################
#   Random Forest 
#####################

class OurRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """A random forest classifier.
    
    Arguments:
        nb_trees --
    
    Attributes:
        base_estimator_ {OurDecisionTreeClassifier} -- The class used to create the list of trees.
        trees_ {list of OurDecisionTreeClassifier} -- The collection of fitted sub-estimators.
        classes_ {set} -- The classes labels.
        n_classes_ {int} -- The number of classes.
        n_features_ {int} -- The number of features when `fit` is performed.
        score_ {float} -- Score of the training dataset (using out-of-bag estimate ??)
    """

    def __init__(self, n_trees = 100, n_samples = 100, n_cuts = 10, max_depth = 15):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.n_cuts = n_cuts
        self.max_depth = max_depth
        self.base_estimator_ = OurDecisionTreeClassifier(n_cuts, max_depth)

    def fit(self, X, y):
        """Build a forest of trees from the training set (X, y).
        
        Arguments:
            X {numpy_ndarray} -- The training input samples without labels
            y {numpy_ndarray} -- The classes labels
        
        Returns:
            OurRandomForestClassifier -- self
        """
        self.classes_ = set(y)
        self.n_classes_ = len(self.classes_)
        self.trees_ = []
        for _ in range(self.n_trees):
        
            # Random sampling of size n_samples in data
            samples = rd.sample(list(X), self.n_samples)

            # Retrieve the labels associated with the data point in samples
            samples_label = [y[k] for k in range(self.n_samples) if (samples[k].all() == X[k].all())]

            # Random sampling of the features that will be used to build the tree 
            self.n_features_ = X.shape[1]
            tFeatures = rd.sample([i for i in range(self.n_features_)], int(sqrt(self.n_features_)) + 1)

            # Build each tree
            tree = self.base_estimator_.fit(X=np.array(samples), y=np.array(samples_label), features=tFeatures)
            self.trees_.append(tree)

        return self

    def predict(self, element):
        """[summary]
        
        Arguments:
            element {numpy-ndarray} -- [description]
        """
        
        occ = [0] * (self.n_classes_ + 1)
        for tree in self.trees_:
            key = tree.predict(element)
            occ[key] += 1
        closest = np.argmax(occ)
        
        return closest

    def score(self, guessed_labels, true_labels):
        """Computes the accuracy of the random forest.
        
        Arguments:
            guessed_labels {list} -- the list of labels guessed by the random forest
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
            
    best_question = (best_feature[0], best_decisional_value)
    
    return best_info, best_c1, best_c2, best_question

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

# Sur les dictionnaires
def lenDictValues(dic):
    """Computes the total number of data in a dataset represented by a dictionary. Each value is a list.
    
    Arguments:
        dic {dict} -- the dataset
    
    Returns:
        int -- the number of data
    """
    s = [len(l) for l in dic.values()]
    return sum(s)


def sampleDict(dic, sample_size):
    """Returns a new dataset of size sample_size with points pick uniformly
    
    Arguments:
        dic {dict} -- the dataset in which we select randomly points
        sample_size {[type]} -- number of points to pick
    
    Returns:
        dict -- the sub dictionary of sample_size values
    """
    idx = rd.sample([i for i in range(lenDictValues(dic))], sample_size)
    d = defaultdict(list)
    for i in idx:
        for k, v in dic.items():
            if i < len(v):
                d[k].append(v[i])
                break
            else:
                i -= len(v)
    return d