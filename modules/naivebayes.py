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
    """A Gaussian Naive Bayes classifier.

    Attributes:
        probaClass {list} -- a list of probalities associated with each class
        meanDf {DataFrame} -- the mean associated with each feature
        sdDf {DataFrame} -- the standard derivation associated with each feature
    """

    def __init__(self):
        self.probaClass = None
        self.meanDf = None
        self.sdDf = None

    def fit(self, train_set):
        """Builds the classifier from the training set.

        Arguments:
            train_set {DataFrame} of shape (n_samples, n_features+1) -- the training input samples with labels

        Returns:
            OurNaivesBayesClassifier -- self
        """
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
        """Predict class for input dataset.

        Arguments:
            dataset {DataFrame} of shape (n_samples, n_features) -- the test input to classify without labels

        Returns:
            list -- the predicted classes
        """

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
        #calculate the score of the prediction
        score = 0
        n = len(guessed_labels)
        for i in range(n):
            score += (int(guessed_labels[i]) == int(true_labels[i]))
        score = score/n

        return score

    def findFeatureImportance(self, dataset):

        feat_imp = {}

        # Make predictions
        y = dataset['genre/label'].values

        dataset = dataset.drop(columns=['genre/label'])
        predictions_without_shuffle = self.predict(dataset)

        # Compute the error value with all columns in place without shuffling
        base = self.score(predictions_without_shuffle, y)

        # Loop through each column
        for ind, col in enumerate(dataset.columns):
            #print("ind =", ind)
            dataset_new = dataset.copy()

            # Shuffle the column
            dataset_new[col] = np.random.permutation(dataset_new[col].values)

            # Make predictions again
            predictions_with_shuffle = self.predict(dataset_new)

            # Compute change in error term as compared to predictions_without_shuffle
            # Greater change means more importance
            feat_imp[ind] = abs(self.score(predictions_with_shuffle, y) - base)

            del dataset_new

        features, importances = zip(*feat_imp.items())
        return features, importances
