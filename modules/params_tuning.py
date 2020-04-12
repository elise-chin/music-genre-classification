# Hyperparameters Tuning
"""
Ce module contient toutes les classes et fonctions permettant d'optimiser les hyperparametres.
"""
import numpy as np
import random as rd

from sklearn.model_selection import ParameterGrid

from modules.scoring import accuracy_score

class RandomizedSearchCV():
    """Randomized search on hyper parameters.
    It implements a `fit` and `predict` methods.

    Arguments:
        estimator {estimator} -- a object of that type is instantiated for each grid point
        param_distributions {dict of list} -- dictionary with parameters names (string) as keys and lists of paramters to try
        n_iter {int} -- number of parameters settings that are sampled
        cv {None, int} -- determines the number of folds in a k-fold cross validation. If None, use the default 5-fold CV

    Attributes:
        cv_results_ {dict of numpy-ndarrays} -- a dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame
        best_estimator_ {estimator} -- estimator which gave highest score 
        best_score_ {float} -- score of the best_estimator
        best_params_ {dict} -- parameter that gave the best results
        best_index_ {int} -- the index of the cv_results_ arrays which corresponds to the best candidate parameter setting
    """
    def __init__(self, estimator, param_distributions, n_iter = 10, cv=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv

    def fit(self, X, y):
        """Run fit with sets of parameters randomly chosen from `self.param_distributions` 
        Each model is evaluated by a `self.cv`-fold cross validation.
        
        Arguments:
            X {numpy_ndarray} of shape (n_samples, n_features) -- the training input samples without labels
            y {numpy_ndarray} of shape (n_samples,) -- the classes labels
        
        Returns:
            RandomizedSearchCV -- self
        """

        self.cv_results_ = {
            "estimators" : [],
            "scores": [],
        }

        # Loop through each iteration
        for _ in range(self.n_iter):

            # Randomly choose a combination of hyper parameters
            params = randomChoiceInDict(self.param_distributions)
            #print("params = ", params)

            # Instanciate the estimator with those hyper parameters
            nth_estimator = self.estimator()
            nth_estimator.set_params(params)
            self.cv_results_["estimators"].append(nth_estimator)

            # Apply k-fold CV
            if(self.cv == None):
                nth_score = kFoldCV(nth_estimator, X, y, 5)
            else:
                nth_score = kFoldCV(nth_estimator, X, y, self.cv)
            
            # Save the nth_estimator score
            self.cv_results_["scores"].append(nth_score)

        # Get the best estimator
        self.best_index_, self.best_score_ = max(enumerate(self.cv_results_["scores"]), key = lambda x:x[1])
        self.best_estimator_ = self.cv_results_["estimators"][self.best_index_]
        self.best_params_ = self.best_estimator_.get_params()

        return self

    def predict(self, X):
        """Best estimator predictions class for X.
        
        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the input to classify
        
        Returns:
            list of int -- the predicted classes
        """
        return self.best_estimator_.predict(X)


class GridSearchCV():
    """Exhaustive search over specified parameter values for an estimator.
    It implements a `fit` and `predict` methods.
    
    Arguments:
        estimator {estimator} -- a object of that type is instantiated for each grid point
        param_distributions {dict of list} -- dictionary with parameters names (string) as keys and lists of paramters to try
        cv {None, int} -- determines the number of folds in a k-fold cross validation. If None, use the default 5-fold CV

    Attributes:
        cv_results_ {dict of numpy-ndarrays} -- a dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame
        best_estimator_ {estimator} -- estimator which gave highest score 
        best_score_ {float} -- score of the best_estimator
        best_params_ {dict} -- parameter that gave the best results
        best_index_ {int} -- the index of the cv_results_ arrays which corresponds to the best candidate parameter setting
    """
    def __init__(self, estimator, param_distributions, n_iter=10, cv=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv

    def fit(self, X, y):
        """Run fit with all combinations of parameters from `self.param_distributions` 
        Each model is evaluated by a `self.cv`-fold cross validation.
        
        Arguments:
            X {numpy_ndarray} of shape (n_samples, n_features) -- the training input samples without labels
            y {numpy_ndarray} of shape (n_samples,) -- the classes labels
        
        Returns:
            GridSearchCV -- self
        """

        self.cv_results_ = {
            "estimators" : [],
            "scores": [],
        }

        params_list = list(ParameterGrid(self.param_distributions))

        # Loop through each combination of parameters
        for params in params_list:
            #print("params = ", params)

            # Instanciate the estimator with those hyper parameters
            nth_estimator = self.estimator()
            nth_estimator.set_params(params)
            self.cv_results_["estimators"].append(nth_estimator)

            # Apply k-fold CV
            if(self.cv == None):
                nth_score = kFoldCV(nth_estimator, X, y, 5)
            else:
                nth_score = kFoldCV(nth_estimator, X, y, self.cv)
            
            # Save the nth_estimator score
            self.cv_results_["scores"].append(nth_score)

        # Get the best estimator
        self.best_index_, self.best_score_ = max(enumerate(self.cv_results_["scores"]), key = lambda x:x[1])
        self.best_estimator_ = self.cv_results_["estimators"][self.best_index_]
        self.best_params_ = self.best_estimator_.get_params()

        return self

    def predict(self, X):
        """Best estimator predictions class for X.
        
        Arguments:
            X {numpy-ndarray} of shape (n_samples, n_features) -- the input to classify
        
        Returns:
            list of int -- the predicted classes
        """
        return self.best_estimator_.predict(X)

def kFoldCV(estimator, X, y, k):
    """K-fold cross validation. It breaks the training data into k equal-sized folds, 
    iterates through each fold, treats that fold as validation set, 
    trains a model on all the other k-1 folds and evaluates the model's performance on the validation fold.
    This results in having k different models, each with an accuracy score.
    The average of these scores is the model's cross-validation score.

    Arguments:
        estimator {estimator} -- the type of the estimator we want to cross-evaluate
        X {numpy_ndarray} of shape (n_samples, n_features) -- the training input samples without labels
        y {numpy_ndarray} of shape (n_samples,) -- the classes labels
        k {int} -- the number of folds
    
    Returns:
        float -- the average of all scores 
    """

    n_samples = X.shape[0]
    n_fold_samples = n_samples//k
    scores = []

    # Split the training set X and the labels y into k subsets
    folds = [X[i*n_fold_samples:(i+1)*n_fold_samples] for i in range(k-1)]
    folds.append(X[(k-1)*n_fold_samples:]) # last fold
    labels_folds = [y[i*n_fold_samples:(i+1)*n_fold_samples] for i in range(k-1)]
    labels_folds.append(y[(k-1)*n_fold_samples:])

    # Loop through each fold
    for kth in range(k):
        #print("kth = ", kth)
        # The kth fold becomes the test subset
        test_subset = folds[kth]
        test_labels_subset = labels_folds[kth]

        # The remaining k-1 folds become the training subset
        training_subset = []
        training_labels_subset = []
        for i in range(k):
            if i != kth:
                training_subset.extend(folds[i])
                training_labels_subset.extend(labels_folds[i])
        
        # Fit the estimator on the training subset
        estimator.fit(np.array(training_subset), np.array(training_labels_subset))
        
        # Evaluate the estimator 
        predictions = estimator.predict(np.array(test_subset))

        # Save the score
        scores.append(accuracy_score(predictions, test_labels_subset))

    return np.mean(scores)


#######################################
# To generate combination of parameters
#######################################
def randomChoiceInDict(dic):
    """Generate a random combination of values
    
    Arguments:
        dic {dict} -- dictionary with multiple values for each key
    
    Returns:
        dict -- dictionary with discrete value for each key
    """
    choice = {}
    for key in dic.keys():
        choice[key] = rd.choice(dic[key])
    return choice



