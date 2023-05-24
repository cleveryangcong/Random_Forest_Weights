'''
Methods for calculating the mean predictions of trees and rf
'''

import numpy as np

def calc_mean_tree(weights_tree, y_train):
    """
        Function that calculates the mean prediction of tree based on the weights of a given X_test input and the y_train sample
    Args:
        weights(multi-dim-array): multi dim array with the weigths for the training sample
        y_train(array): Array with y values used for training of the specific tree
    Returns:
        array: array of mean values
    """
    X_test_length = len(weights_tree)
    means = []
    for i in range(X_test_length):
        mean = np.dot(weights_tree[i], y_train)
        means.append(mean)
    return np.array(means)


def calc_mean_rf(weights_rf, y_train):
    """
        Function that calculates the mean prediction of rf based on the weights of a given X_test input and the y_train sample
    Args:
        weights_rf(3-dim-array): three dimensional array with the weigths for the training sample
        y_train(array): Array with y values used for training of the specific tree
    Returns:
        array: array of mean values

    """
    number_of_trees = len(weights_rf)
    X_test_length = len(weights_rf[0])
    tree_means = []
    for i in range(number_of_trees):
        tree_mean = calc_mean_tree(weights_rf[i], y_train)
        tree_means.append(tree_mean)
    means = [np.mean(sublist) for sublist in zip(*tree_means)]
    return np.array(means)