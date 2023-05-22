import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split



def calc_obs_distribution(tree, X_train, y_train):
    """
    Function to calculate the distribution of observations in the leaf nodes of a tree-model
    Input:
        param tree: Fitted DecisionTreeRegressor
        param X_train: Training samples used for fitting
        param y_train: Training observations
    Output:
        leaf_nodes: Ordered numerical array with index of leaf nodes
        y_values_leaves: List of arrays with y_values that fall into each leaf node (same order as leaf_nodes)
        X_values_leaves: List of arrays with X_values that fall into each leaf node (same order as leaf_nodes)
        X_ids_leaves: List of arrays that determine whether a training input is in a leaf or not.
                      Length of number of leaves
        weights_leaves: List of arrays with the weights of the training obs dependent on leaf

    """
    # Calculate index of the leaf that each sample is predicted as
    leaf_id_train = tree.apply(X_train)

    # Get ids of leaf sorted
    leaf_nodes = np.unique(leaf_id_train)

    # Get list of boolean arrays with information on what observation is in which leaf
    X_ids_leaves = []
    for i in leaf_nodes:
        X_ids_leaves.append(leaf_id_train == i)

    # Get list of what observation values are in which leaf
    y_values_leaves = []
    for i in range(len(X_ids_leaves)):
        y_values_leaves.append(y_train[X_ids_leaves[i]].flatten())

    # Get list of what input values are in which leaf
    X_values_leaves = []
    for i in range(len(X_ids_leaves)):
        X_values_leaves.append(X_train[X_ids_leaves[i]])

    # Get list of weights of train obs of each leaf
    weights_leaves = []
    for i in range(len(X_ids_leaves)):
        weights_true = 1 / np.sum(X_ids_leaves[i])
        help_matrix = X_ids_leaves[i].astype(int)
        help_matrix = help_matrix.astype(float)
        help_matrix[help_matrix == 1] = weights_true
        weights_leaves.append(help_matrix)

    return leaf_nodes, y_values_leaves, X_values_leaves, X_ids_leaves, weights_leaves



def calc_mean_observation_tree(tree, X_test, y_train, leaf_nodes, weights_leaves):
    """
    Calculate weights of a tree and mean prediction based on a Out-Of_Sample Dataset

    input:
        param tree: Fitted DecisionTreeRegressor
        param X_test: OOS Dataset
        param y_train: observations used to build tree
        param leaf_nodes: List of length number_leaf_nodes with arrays with leaf_node indexes
        param weights_leaves: List of length number_leaf_nodes with weights of individual leaf nodes

    output:
        mean_preds: List of mean predictions of tree
        weights: List of weights used to calculate mean_preds
    """
    #  Calculate index of the leaf that each sample is predicted as
    X_test_id_leaves = tree.apply(X_test)

    mean_preds = []
    weights = []
    for i in range(len(X_test_id_leaves)):
        X_id = X_test_id_leaves[i]
        index = np.where(leaf_nodes == X_id)[0][0]  # Calculate index of test

        # Calculate mean through sum(weights * y_train)
        weights.append(weights_leaves[index])
        mean = np.dot(weights_leaves[index], y_train)[0]
        mean_preds.append(mean)

    return mean_preds, weights



def calc_weights_rf(rf, X_test, y_train, leaf_nodes_trees, weights_leaves_trees):
    '''
    Method to calculate the mean prediction and weights of a random forest
    
    Input: 
        param rf: Fully fitted random Forest
        param X_test: OOS test data
        param y_train: Data used to train the RF
        param leaf_nodes_trees: 3-Dimensional: 1. number_trees, 2. number_leaf_nodes, 
                                3. array with leaf node indexes
        param weights_leaves_trees: 3-Dimensional: 1. number_trees, 2. number_leaf_nodes
                                    3. array with weights of individual leaf_nodes
    
    Output: 
        weights_all: list of length X_test with weights used to calculate mean prediction
        mean_preds: List of mean predictions
    '''
    # Calculate index of the leaf that each sample is predicted as in all trees
    X_test_id_leaves = []  # dim: num_trees x len_X_test
    for tree in rf.estimators_:  # iterate number of tree times
        X_test_id_leaves.append(tree.apply(X_test))

    weights_all = []
    mean_preds = []
    for i in range(len(X_test)):  # iterate number of X_test times
        weight_k = np.zeros(y_train.shape)
        for j in range(len(X_test_id_leaves)):  # iterate number of trees times
            X_id = X_test_id_leaves[j][i]
            index = np.where(leaf_nodes_trees[j] == X_id)[0][
                0
            ]  # Calculate index of test
            weight_k = weight_k + weights_leaves_trees[j][index]
        weight = weight_k / len(X_test_id_leaves)
        weights_all.append(weight)
        mean_preds.append(np.dot(weight, y_train))

    return weights_all, mean_preds