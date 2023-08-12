'''
Methods for calculating the weights of regression trees and regression random forests
'''
#Basics:
from collections import Counter
import numpy as np
import sklearn.ensemble._forest as forest_utils
from tqdm import tqdm



def calc_weights_tree(tree, X_train_base, X_test, bootstrap, max_samples):
    """
        Calculates weights of not bootstrapped training sample of a fitted (bootstrapped) tree
        Only works for random forest trees (random values)
    Args:
        tree(decisiontree): fitted DecisionTreeRegressor on X_train_boot
        X_train_base(Array): base training sample
        X_test(array): test sample
        bootstrap(boolean): whether tree was trained with bootstraping
        max_samples(float or int): if bootstrap == True, number of samples to draw from X to train
    Returns:
        array: length(X_train_base) with weights of the base training sample
    """
    n_samples = X_train_base.shape[0]
    n_samples_bootstrap = forest_utils._get_n_samples_bootstrap(n_samples, max_samples)

    unsampled_indices = forest_utils._generate_unsampled_indices(
        tree.random_state, n_samples, n_samples_bootstrap
    )
    sampled_indices = forest_utils._generate_sample_indices(
        tree.random_state, n_samples, n_samples_bootstrap
    )

    if bootstrap:
        X_train = X_train_base[sampled_indices]
    else:
        X_train = X_train_base

    leaf_id_train = tree.apply(X_train)
    leaf_nodes = np.unique(leaf_id_train)
    
    sampled_counts = Counter(sampled_indices)
    sample_counts_dict = defaultdict(int, {i: sampled_counts.get(i, 0) for i in range(n_samples)})

    # Instead of looping through leaves and using multiple lists, use matrices directly
    num_leaves = len(leaf_nodes)
    leaf_matrix = np.zeros((num_leaves, X_train.shape[0]))
    for idx, leaf in enumerate(leaf_nodes):
        leaf_matrix[idx] = (leaf_id_train == leaf) / np.sum(leaf_id_train == leaf)

    X_test_id_leaves = tree.apply(X_test)
    weights = np.array([leaf_matrix[np.where(leaf_nodes == leaf_id)[0][0]] for leaf_id in X_test_id_leaves])
    
    if not bootstrap:
        return weights
    else:
        final_weights = np.zeros((len(X_test_id_leaves), X_train_base.shape[0]))
        for i in range(len(X_test_id_leaves)):
            total_length = sampled_indices[weights[i] > 0].shape[0]
            unique_base_indices = np.unique(sampled_indices[weights[i] > 0])
            for base_idx in unique_base_indices:
                bootstrapped_times = sample_counts_dict[base_idx]
                final_weights[i, base_idx] = bootstrapped_times / total_length
        return final_weights



def calc_weights_rf(rf, X_train_base, X_test, bootstrap, max_samples):
    """
        Calculates weights of a random forest
    Args:
        rf(RandomForestRegressor): Trained random forest
        X_train_base(array): Training sample used to train random forest
        X_test(array): Test sample to evaluate on
        bootstrap(boolean): whether bootstrapping is used or not
        max_samples(float or int): if bootstrap == True, number of samples to draw from X to train
    Returns:
        array: 3-dim-array with weights of rf shape (num_trees, num_X_test, num_leaves)
    """
    num_trees = len(rf.estimators_)
    # Assuming the shape of the weights from a single tree will always be (num_X_test, num_leaves), initializing with zeros
    weights_rf = np.zeros((num_trees, X_test.shape[0], X_train_base.shape[0]))

    for idx, tree in tqdm(enumerate(rf.estimators_)):
        weights_rf[idx] = calc_weights_tree(tree, X_train_base, X_test, bootstrap, max_samples)

    return weights_rf