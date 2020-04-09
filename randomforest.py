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

class OurDecisionTreeClassifier():
    """A decision tree classifier.
    
    Arguments:
        n_cuts {int} -- the number of cuts tested to find the best cut
        max_depth {int} -- the maximal depth of a decision tree
        
    Attributes:
        tree_ {Node} -- the root of the decision tree
        n_features_ {int} -- the total number of features in the training set
        features_ {list} -- the list of features randomly chosen when `fit` is performed
        max_features_ {int} -- the number of features when `fit` is performed (attention <= nombre de features dans dataset)
    """
    def __init__(self, n_cuts = 40, max_depth = 20):
        self.n_cuts = n_cuts
        self.max_depth = max_depth

    def fit(self, node_train, features=None):
        """Builds the decision tree from the training set (node_train).
        
        Arguments:
            node_train {Node} -- the node which contains the training set in dictionnary form
        
        Keyword Arguments:
            features {list} -- the list of features (default: {None})
        
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

        self.n_features_ = len(list(node_train.data.values())[0][0])
        self.max_features_ = int(sqrt(self.n_features_) + 1)

        if(features == None):
            self.features_ = rd.sample([i for i in range(self.n_features_)], self.max_features_)
        else:
            self.features_ = features

        # Build the tree
        self.tree_ = decisionTree(node=node_train, n_cuts=self.n_cuts, features=self.features_, max_depth=self.max_depth)

        return self

    def predict(self, element):
        """Predict class for X.
    
        Arguments:
            element {numpy-ndarray} of shape (n_features,) -- the input to classify
        
        Returns:
            list of int -- the predicted classes
        """
        
        node = self.tree_
        exists_son = True
        while(node != None and node.question != None and exists_son):
            if(element[node.question[0]] <= node.question[1]):
                node = node.left_son
            else:
                node = node.right_son
            if(node.left_son == None and node.right_son == None):
                exists_son = False
        majoritary_index = np.argmax([len(points) for points in list(node.data.values())])
        key = list(node.data.keys())[majoritary_index]
        return key

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

#####################
#   Random Forest 
#####################

class OurRandomForestClassifier():
    """A random forest classifier.
    
    Arguments:
        n_trees {int} -- the number of trees in the forest
        n_samples {int} -- the number of data points placed in a node before the node is split
        n_cuts {int} -- the number of cuts tested to find the best cut
        max_depth {int} -- the maximal depth of each tree
    
    Attributes:
        base_estimator_ {OurDecisionTreeClassifier} -- the class used to create the list of trees
        trees_ {list of OurDecisionTreeClassifier} -- the collection of fitted sub-estimators
        n_classes_ {int} -- the number of classes of the training set
    """

    def __init__(self, n_trees = 100, n_samples = 100, n_cuts = 20, max_depth = 20):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.n_cuts = n_cuts
        self.max_depth = max_depth
        self.base_estimator_ = OurDecisionTreeClassifier(n_cuts, max_depth)

    def fit(self, data_train_dict):
        """Build a forest of trees from the training set (data_train_dict).
        
        Arguments:
            data_train_dict {dict} -- the training input samples contained in a dictionary
        
        Returns:
            OurRandomForestClassifier -- self
        """
        self.n_classes_ = len(data_train_dict.keys())
        self.trees_ = []
        for _ in range(1, self.n_trees + 1):
            # Initialisation of the decision tree
            dt = OurDecisionTreeClassifier(self.n_cuts, self.max_depth)

            # Random sampling of size sample_size in data
            d = sampleDict(data_train_dict, self.n_samples)
            root = Node(d)
            
            # Random sampling of the features that will be used to build the tree 
            nb_features = len(list(data_train_dict.values())[0][0])
            tFeatures = rd.sample([i for i in range(0,nb_features)], int(sqrt(nb_features)) + 1)
            dt.fit(root, tFeatures)
            self.trees_.append(dt)

        return self

    def predict(self, element):
        """Predict class for element.
        
        Arguments:
            element {numpy-ndarray} of shape (n_features,) -- the input to classify
        
        Returns:
            list of int -- the predicted classes
        """
        occ = [0] * (self.n_classes_ + 1)
        for tree in self.trees_:
            key = tree.predict(element)
            occ[key] += 1
        return np.argmax(occ)

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

    def findFeatureImportance(self, X, y):
        """Calculate feature importances according to 
        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the training dataset without labels
            y {numpy-ndarray} of shape (n_samples,) -- the labels of the training dataset
        Returns:
        """

        n_samples, n_features = X.shape[0], X.shape[1]
        feat_imp = {}

        # Make predictions
        predictions_without_shuffle = [self.predict(X[i,:]) for i in range(n_samples)]

        # Compute the error value with all columns in place without shuffling
        base = self.score(predictions_without_shuffle, y)

        # Loop through each column 
        for ind in range(n_features):
            #print("ind = ", ind)
            X_new = X.copy()

            # Shuffle the column
            np.random.shuffle(X_new[:, ind])

            # Make predictions again
            predictions_with_shuffle = [self.predict(X_new[i,:]) for i in range(n_samples)]

            # Compute change in error term as compared to predictions_without_shuffle
            # Greater change means more importance 
            feat_imp[ind] = abs(self.score(predictions_with_shuffle, y) - base)

            del X_new

        features, importances = zip(*feat_imp.items())
        return features, importances

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
            
    #best_question = (best_feature[0], best_decisional_value)
    
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

####################
# Feature Selection
####################

def transform(X, selected_features_indexes):
    """Reduce X to the selected features.
    
    Arguments:
        X {numpy-ndarray} of shape (n_samples, n_features) -- the input samples
        selected_features_indexes {list} -- the indexes of the selected features
    
    Returns:
        numpy-ndarray -- the input samples with only the selected features
    """
    X_r = np.array([X[:,i] for i in selected_features_indexes])
    return np.transpose(X_r)
