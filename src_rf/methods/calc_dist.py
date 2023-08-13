from scipy.sparse import csr_matrix
from scipy.optimize import bisect
import numpy as np
def calc_dist_rf(weights_rf, y_train):
    """
    Function that calculates the empirical cumulative distribution of a rf based on the weights of a given X_test input and the y_train sample
    Args:
        weights_rf(list[csr_matrix]): List of sparse matrices representing weights for each tree in the forest
        y_train(array): Array with y values used for training of the specific tree
    Returns:
        list of fcts: empirical distribution functions
    """
    
    # Averaging across trees
    avg_weights = sum(weights for weights in weights_rf) / len(weights_rf)
    
    rf_cdfs = []
    for i in range(avg_weights.shape[0]):

        def cdf_func(y, i=i):  # Capture the current value of i
            row = avg_weights.getrow(i).toarray().squeeze()
            cdf_sum = sum(w for point, w in zip(y_train, row) if point <= y)
            return np.round(cdf_sum, 4)

        rf_cdfs.append(cdf_func)
    return rf_cdfs

def calc_quantile_rf(cdfs, q, y_train):
    """
    Estimate the q-th quantiles of the distribution defined by the given CDFs.
    Args:
        cdfs (list): The CDFs, list of functions.
        q(float): The desired quantile, a number between 0 and 1.
        y_train(array): The training data, a 1D numpy array.
    Returns:
        list of fcts: The estimated q-th quantiles.
    """
    quantiles = []
    for i in range(len(cdfs)):
        # Define a function that is zero when cdf(y) = q
        func = lambda y: cdfs[i](y) - q

        # Use the bisection method to find the root of this function
        quantiles.append(bisect(func, min(y_train), max(y_train)))

    return quantiles
