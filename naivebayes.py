import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from math import sqrt
from math import pi
from math import exp
from math import log

class OurNaiveBayesClassifier():
    
    def __init__(self):
        self.probaClass = None
        self.meanDf = None
        self.sdDf = None
    
    def fit(self, train_set):
        train_set_by_classes = train_set.groupby('genre/label')
        
        framesMean = []
        framesSd = []
        probaClass = np.empty([1, len(train_set_by_classes)])
        sizeSample = len(train_set)
        for i in range(len(train_set_by_classes)):
            group = train_set_by_classes.get_group(i)
            probaClass[0,i] = len(group)/sizeSample
            meanFeaturesClass = pd.DataFrame(group.mean(axis=0)).rename(columns={0:i})
            sdFeaturesClass = pd.DataFrame(group.std(axis = 0)).rename(columns={0:i})
            framesMean.append(meanFeaturesClass)
            framesSd.append(sdFeaturesClass)
            
        meanDf = pd.concat(framesMean, axis=1, join='inner')
        sdDf = pd.concat(framesSd, axis=1, join='inner')
        
        meanDf = meanDf.transpose().drop(axis=1, labels='genre/label')
        sdDf = sdDf.transpose().drop(axis=1, labels='genre/label')
        
        self.probaClass = probaClass
        self.meanDf = meanDf
        self.sdDf = sdDf
        return self

    def predict(self, dataset):
        
        # Calculate the Gaussian probability distribution function for x
        def calculate_proba_Norm(x, mean, stdev):
        	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        	return (1 / (sqrt(2 * pi) * stdev)) * exponent
        
        def calculateProbabilityClass(label, indexClass, probaClass, meanDf, sdDf):
            value = float(0)
            valExp = exp(1)
            for i in range(label.size):
                value += log(calculate_proba_Norm(label[i], meanDf.iat[indexClass, i], sdDf.iat[indexClass, i]) + 1, valExp) #on rajoute un pour eviter d'avoir une proba tres faible
            value +=  log(probaClass[0,indexClass], valExp)
            return value
        
        predictions = []
        for index, row in dataset.iterrows():
            li = [calculateProbabilityClass(row, i, self.probaClass, self.meanDf, self.sdDf) for i in range(len(self.meanDf))]
            predictions.append(np.argmax(np.asarray(li)))
            
        return predictions

    def score(self, guessed_labels, true_labels):
        score = 0
        n = len(guessed_labels)
        for i in range(n):
            score += (int(guessed_labels[i]) == int(true_labels[i]))
        score = score/n
        
        return score
    
    def findFeatureImportance(self, dataset):
        """Calculate feature importances according to score function of OurDecisionTreeClassifier.

        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the training dataset without labels
            y {numpy-ndarray} of shape (n_samples,) -- the labels of the training dataset

        Returns:
            tuple of list -- the features associated with their importance
        """

        feat_imp = {}

        # Make predictions
        y = dataset['genre/label'].to_list()
        
        dataset = dataset.drop(columns=['genre/label'])
        n_features = len(dataset.columns)
        predictions_without_shuffle = self.predict(dataset)

        # Compute the error value with all columns in place without shuffling
        base = self.score(predictions_without_shuffle, y)

        # Loop through each column
        for ind in range(n_features): #pb lors de la derniere iteration
            if ind>=1: break;
            print("ind = ", ind)
            dataset_new = dataset.copy()
            print(dataset_new.head(10))

            # Shuffle the column JE N'ARRIVE PAS A SHUFFLE UNE COLONNE
            #np.random.shuffle(X_new[:, ind])
            dataset_new.reset_index(drop=True, inplace=True) 
            
            dataset_shuffle = dataset_new.sample(frac=1)
            serie = dataset_shuffle.iloc[:,ind]
            
            dataset_new.update(serie)
            
            print(dataset_new.head(10))
            # Make predictions again
            predictions_with_shuffle = self.predict(dataset_new)

            # Compute change in error term as compared to predictions_without_shuffle
            # Greater change means more importance 
            feat_imp[ind] = abs(self.score(predictions_with_shuffle, y) - base)

            del dataset_new

        features, importances = zip(*feat_imp.items())
        return features, importances

