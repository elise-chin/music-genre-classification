import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from math import sqrt
from collections import defaultdict

#######################
#    Decision Tree
#######################

class Node:
    def __init__(self, data):
        self.father = None
        self.leftSon = None
        self.rightSon = None
        self.data = data
        self.question = None
    
    def displayTree(self, depth):
        for i in range(depth):
            if(i < depth - 1):
                print("   ", end = " ")
            if(i == depth - 1):
                print("|-->", end = " ")
        if(self.question == None):
            print("None")
        else:
            print("({0[0]}, {0[1]:.2f})".format(self.question))
        if(self.leftSon != None):
            self.leftSon.displayTree(depth + 1)
        if(self.rightSon != None):
            self.rightSon.displayTree(depth + 1)
        
    def depth(self):
        if self.rightSon == None and self.leftSon == None:
            return 0
        else:
            maxLeft = 0
            maxRight = 0
            if(self.leftSon != None):
                dLeft = self.leftSon.depth()
                if dLeft > maxLeft :
                    maxLeft = dLeft
            if(self.rightSon != None):
                dRight = self.rightSon.depth()
                if dRight > maxRight:
                    maxRight = dRight
            if maxLeft >= maxRight:
                m = maxLeft
            else:
                m = maxRight
            return m + 1

def gini(dic):
    probs = [len(L) for L in dic.values()]
    probs2 = []
    s = sum(probs)
    for prob in probs:
        probs2.append(prob/s)
    #probs2 = [prob/s for prob in probs]
    return 1 - sum([i ** 2 for i in probs2])

def lenDictValues(dic):
    s = [len(l) for l in dic.values()]
    return sum(s)
    
def infoGain(c, c1, c2):
    p = float(lenDictValues(c1)/lenDictValues(c))
    return gini(c) - (p*gini(c1) + (1-p)*gini(c2))

def generateCut(node, features):
    # Feature random choice.
    randomFeature = rd.sample(features, 1)
    
    points = []
    values = node.data.values()
    c1 = defaultdict(list)
    c2 = defaultdict(list)
    decisionalValue = 0
    
    if len(values) != 0:
        # Put all the values in the list points ...
        for line in node.data.values():
            points.extend(line)
        
        # ... in order to randomly choose the decisional value
        decisionalValue = float(rd.choice([v[randomFeature] for v in points]))
        
        # Construct the two datasets stem from the cut.
        for key, l in node.data.items():
            for v in l:
                if v[randomFeature] < decisionalValue:
                    c1[key].append(v)
                else:
                    c2[key].append(v)
                    
    return c1, c2, randomFeature, decisionalValue

def bestCut(node, nbCut, features):
    bestC1, bestC2, bestFeature, bestDecisionalValue = generateCut(node, features) # keep track of the split
    bestInfo = infoGain(node.data, bestC1, bestC2) # keep track of the best information gain.
    
    for _ in range(nbCut-1):
        c1, c2, RD, DV = generateCut(node, features)
        
        # Skip the cut if the dataset was not divided.
        if len(c1) == 0 or len(c2) == 0:
            continue
        
        info = infoGain(node.data, c1, c2)
        if(info > bestInfo):
            bestInfo = info
            bestC1, bestC2, bestFeature, bestDecisionalValue = c1, c2, RD, DV
            
    bestQuestion = (bestFeature[0], bestDecisionalValue)
    
    return bestInfo, bestC1, bestC2, bestQuestion

def partition(node, c1, c2, question):
    node.question = question
    node.leftSon = Node(c1)
    node.leftSon.father = node
    node.rightSon = Node(c2)
    node.rightSon.father = node

def decisionTree(node, nb_cut_to_find_best, features, maxDepth=5):
    gain, c1, c2, question = bestCut(node, nb_cut_to_find_best, features)
    
    # There is no further information gain.
    if(gini(node.data) == 0 or gain == 0 or node.depth() == maxDepth):
        return node # Leaf
    
    # The best cut has been found. We can then partition the dataset.
    partition(node, c1, c2, question)
    
    # Recursively builds the tree.
    decisionTree(node.leftSon, nb_cut_to_find_best, features, maxDepth-1)
    decisionTree(node.rightSon, nb_cut_to_find_best, features, maxDepth-1)
    return node # Decision node

def classify(root, x, numFig):
    node = root
    existsSon = True
    while(node != None and node.question != None and existsSon):
        if(x[node.question[0]] <= node.question[1]):
            node = node.leftSon
        else:
            node = node.rightSon
        if(node.leftSon == None and node.rightSon == None):
            existsSon = False
    majoritary_index = np.argmax([len(points) for points in list(node.data.values())])
    key = list(node.data.keys())[majoritary_index]
    return int(key)

#####################
#   Random Forest 
#####################

def sampleDict(dic, nbObs):
    idx = rd.sample([i for i in range(lenDictValues(dic))], nbObs)
    d = defaultdict(list)
    for i in idx:
        for k, v in dic.items():
            if i < len(v):
                d[k].append(v[i])
                break
            else:
                i -= len(v)
    return d

def randomForest(data, nbTrees, sampleSize, nb_cut_to_find_best, maxDepth = 5, verbose = False):
    listTrees = []
    for i in range(1, nbTrees + 1):
        
        # Random sampling of size sampleSize in data
        d = sampleDict(data, sampleSize)
        root = Node(d)
        
        # Random sampling of the features that will be used to build the tree 
        nbFeatures = len(list(data.values())[0][0])
        tFeatures = rd.sample([i for i in range(0,nbFeatures)], int(sqrt(nbFeatures)) + 1)
        decisionTree(root, nb_cut_to_find_best, tFeatures, maxDepth)
        listTrees.append(root)
        
        if verbose == True:
            print('tFeatures : ', tFeatures)
            print("Tree", i)
            root.displayTree(0)
        
    return listTrees    

def forestClassify(listTrees, x, nbLabels):
    occ = [0] * (nbLabels + 1)
    for numFig, tree in enumerate(listTrees):
        key = classify(tree, x, numFig)
        occ[key] += 1
    return np.argmax(occ)


def score(guessedlabels,truelabels):
    score = 0
    n = len(guessedlabels)
    for i in range(n):
        score += (int(guessedlabels[i]) == int(truelabels[i]))
    score = score/n
    return score