'''
Methods for calculating the empirical dists and quantile predictions of rfs
'''

import numpy as np
from scipy.optimize import bisect


def calc_dist_rf(weights_rf, y_train):
    """
    Function that calculates the empirical cumulative distribution of a rf based on the weights of a given X_test input and the y_train sample
    Args:
        weights_rf(3-dim-array): three dimensional array with the weigths for the training sample
        y_train(array): Array with y values used for training of the specific tree
    Returns:
        list of fcts: empirical distribution functions

    """
    number_of_trees = weights_rf.shape[0]
    X_test_length = weights_rf.shape[1]
    weights_rf_reduced = np.mean(weights_rf, axis=0)
    rf_cdfs = []

    for i in range(X_test_length):

        def cdf_func(y, i=i):  # Capture the current value of i
            cdf_sum = sum(
                w for point, w in zip(y_train, weights_rf_reduced[i]) if point <= y
            )
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