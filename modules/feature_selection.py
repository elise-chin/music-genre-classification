# Feature selection

import numpy as np
import modules.scoring as sc

def findFeatureImportance(estimator, X, y):
    """Calculate feature importances. Wrapper-based method.

    Arguments:
        X {numpy-ndarray} of shape (n_samples, n_features) -- the training dataset without labels
        y {numpy-ndarray} of shape (n_samples,) -- the labels of the training dataset

    Returns:
        tuple of list -- the features associated with their importance
    """

    n_features = X.shape[1]
    feat_imp = {}

    # Make predictions
    predictions_without_shuffle = estimator.predict(X)

    # Compute the error value with all columns in place without shuffling
    base = sc.accuracy_score(predictions_without_shuffle, y)

    # Loop through each column 
    for ind in range(n_features):
        #print("ind = ", ind)
        X_new = X.copy()

        # Shuffle the column
        np.random.shuffle(X_new[:, ind])

        # Make predictions again
        predictions_with_shuffle = estimator.predict(X_new)

        # Compute change in error term as compared to predictions_without_shuffle
        # Greater change means more importance 
        feat_imp[ind] = abs(sc.accuracy_score(predictions_with_shuffle, y) - base)

        del X_new

    features, importances = zip(*feat_imp.items())
    return features, importances

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