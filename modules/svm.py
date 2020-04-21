import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Scikit-kearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class OurSVMClassifier():
    """Classifier implementing SVM without Kernel Trick.

    Arguments:
        reg_strength {int} -- regularization constant
        max_iter {int} -- number of iteration for the stochastic gradient descent
        batch_size {int} -- number of points use by each iteration of the stochastic gradient descent
        learning_rate {int} -- size of each step of the stochastic gradient descent

    Attributes:
        w {numpy-ndarray} of shape (n_features) -- the weigths of the prediction model for describing the hyperplan
        b {int} of shape (n_features) -- constant of the prediction model for describing the hyperplan
    """
    def __init__(self, reg_strength, max_iter, batch_size, learning_rate):
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        Arguments:
            X {numpy_ndarray} of shape (n_samples, n_features) -- the training input samples without labels
            y {numpy_ndarray} of shape (n_samples,) -- the classes labels

        Returns:
            OurSVMClassifier -- self
        """

        #for calculate the score function we try to minimize giwen w and b
        def scoreFunction(w, b, data_train, label_train, reg_strength):
            summ = 0
            for i in range(data_train.shape[0]):
                summ += max(0, 1 - label_train[i] * (np.dot(np.transpose(w),data_train[i,:]) + b))
            return (1/2)*np.dot(np.transpose(w),w) + reg_strength * summ

        #for calculate the gradient of the score function
        def gradient(w, batch, reg_strength):

            nb_features = batch.shape[1] - 1
            summ = np.ones((nb_features+1,1))

            for i in range(batch.shape[0]):
                yi = batch[i, -1]
                xi = batch[i, :]

                xi = xi[:-1]
                xi = np.append(xi, 1)
                xi = np.reshape(xi, (nb_features+1,1))

                if(yi*np.dot(np.transpose(w),xi) <= 1):
                    #xi[-1] = 0
                    summ += -(yi * xi)

            return w + (reg_strength * summ)

        data = np.c_[X, y]
        data = np.asarray(data)
        nb_features = data.shape[1] - 1
        w = np.ones((nb_features +1,1))

        L = []

        #stochastic gradient descent
        for itr in range(self.max_iter):
            random_indices = np.arange(0, data.shape[0])
            np.random.shuffle(random_indices)
            w = np.subtract(w, self.learning_rate * gradient(w, data[random_indices[:self.batch_size]], self.reg_strength))
            L.append((scoreFunction(w[:-1], w[-1], X, y, self.reg_strength),itr))

        plt.plot(L)
        self.w = w[:-1]
        self.b = w[-1]
        #print(scoreFunction(self.w,self.b, X, y, self.reg_strength))
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
            val = np.sign(np.dot(np.transpose(self.w), X[i,:]) + self.b)[0]
            predictions.append(val)
        return predictions

class OurMultiClassOAASVMClassifier():
    """Classifier implementing SVM One Against All.

    Arguments:
        reg_strength {int} -- regularization constant
        max_iter {int} -- number of iteration for the stochastic gradient descent
        batch_size {int} -- number of points use by each iteration of the stochastic gradient descent
        learning_rate {int} -- size of each step of the stochastic gradient descent

    Attributes:
        List_SVM {list} of size (nb_classes) -- a list of SVM Classifier, one SVM for one class
    """
    def __init__(self, reg_strength = 50, max_iter = 100, batch_size = 200, learning_rate = 0.0001):
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def get_params(self):
        """Get the parameters of the SVM OAA classifier.

        Returns:
            dict -- estimator parameters
        """
        params = {
            'reg_strength': self.reg_strength,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
        }
        return params

    def set_params(self, params):
        """Set the parameters of OAA SVM.

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
            OurOAASVMClassifier -- self
        """
        nb_classes = np.max(y)
        List_SVM = []
        for label_number in range(nb_classes):
            base_model = OurSVMClassifier(self.reg_strength, self.max_iter, self.batch_size, self.learning_rate)
            new_label = (y == label_number).astype(int)
            new_label[new_label == 0] = -1

            base_model.fit(X, new_label)
            List_SVM.append(base_model)

        self.List_SVM = List_SVM

    def predict(self, X):
        """Predict class for X.

        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the test input to classify

        Returns:
            list -- the predicted classes
        """
        predictions = []
        for i in range(X.shape[0]):
            list_value_decison_function = []
            for svm in self.List_SVM:
                list_value_decison_function.append(np.dot(np.transpose(svm.w), X[i,:]) + svm.b) #we choose the the class with the highest margin svm
            predictions.append(np.argmax(list_value_decison_function))
        return predictions

class OurMultiClassOVOSVMClassifier():
    """Classifier implementing SVM One Against All.

    Arguments:
        reg_strength {int} -- regularization constant
        max_iter {int} -- number of iteration for the stochastic gradient descent
        batch_size {int} -- number of points use by each iteration of the stochastic gradient descent
        learning_rate {int} -- size of each step of the stochastic gradient descent

    Attributes:
        List_SVM {list} of size (nb_classes) -- a list of SVM Classifier, one for each combinations of two classes
        list_perm {list} of size ()(nb_classes*(nb_classes-1))/2) -- a list of all the comninations with the given classes
    """
    def __init__(self, reg_strength = 100, max_iter = 40, batch_size = 16, learning_rate = 0.0001):
        self.reg_strength = reg_strength
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def get_params(self):
        """Get the parameters of the SVM OAA classifier.

        Returns:
            dict -- estimator parameters
        """
        params = {
            'reg_strength': self.reg_strength,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
        }
        return params

    def set_params(self, params):
        """Set the parameters of OAA SVM.

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
            OurOVOSVMClassifier -- self
        """
        nb_classes = np.max(y)
        data = np.c_[X, y]

        li = [i for i in range(nb_classes + 1)]
        list_perm = itertools.combinations_with_replacement(li, 2)
        list_perm = [tup for tup in list_perm if tup[0] != tup[1]]

        List_SVM = []

        for comb in list_perm:

            dataIte = [x for x in data if (x[-1] == comb[0] or x[-1] == comb[1])]
            X_ite, y_ite = tuple(np.split(dataIte, [-1], axis=1))

            base_model = OurSVMClassifier(self.reg_strength, self.max_iter, self.batch_size, self.learning_rate)

            new_label = (y_ite == comb[0]).astype(int)
            new_label[new_label == 0] = -1

            base_model.fit(X_ite, new_label)
            List_SVM.append(base_model)

        self.List_SVM = List_SVM
        self.list_perm = list_perm

    def predict(self, X):
        """Predict class for X.

        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the test input to classify

        Returns:
            list -- the predicted classes
        """
        predictions = []
        #we proceed to a vote with all the SVM classifiers
        for i in range(X.shape[0]):
            votes = [0 for i in range(self.list_perm[-1][1] + 1)]
            for svm, perm in zip(self.List_SVM, self.list_perm):
                val = np.sign(np.dot(np.transpose(svm.w), X[i,:]) + svm.b)[0]
                if val > 0:
                    votes[perm[0]] += 1
                else:
                    votes[perm[1]] += 1
            predictions.append(np.argmax(votes))
        return predictions
