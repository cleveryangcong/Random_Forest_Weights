import sklearn.ensemble._forest as forest_utils
from collections import Counter


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
    # 1. Get number of times a value from base X_train_base is sampled in the bootstrap sample X_train_boot
    n_samples = X_train_base.shape[0]

    n_samples_bootstrap = forest_utils._get_n_samples_bootstrap(n_samples, max_samples)

    unsampled_indices = forest_utils._generate_unsampled_indices(
        tree.random_state, n_samples, n_samples_bootstrap
    )
    sampled_indices = forest_utils._generate_sample_indices(
        tree.random_state, n_samples, n_samples_bootstrap
    )
    # Get X_train_bootstrap
    X_train_bootstrap = X_train_base[sampled_indices]

    # count the number of occurrences of each index in the bootstrap sample
    sampled_counts = Counter(sampled_indices)

    # create a dictionary where keys are the indices of the original samples and values are the number of times each sample was included in the bootstrap sample
    sample_counts_dict = {i: sampled_counts.get(i, 0) for i in range(n_samples)}

    # Decide based on bootstrap which dataset tree is trained on:
    if bootstrap:
        X_train = X_train_bootstrap
    else:
        X_train = X_train_base

    # 2. Find all leaves of a tree
    # Calculate index of the leaf that each sample is predicted as
    leaf_id_train = tree.apply(X_train)

    # Get ids of leaf sorted
    leaf_nodes = np.unique(leaf_id_train)

    # Get list of boolean arrays with information on what observation is in which leaf
    X_ids_leaves = []
    for i in leaf_nodes:
        X_ids_leaves.append(leaf_id_train == i)

    # Get list of weights of train obs of each leaf
    weights_leaves = []
    for i in range(len(X_ids_leaves)):
        weights_true = 1 / np.sum(X_ids_leaves[i])
        help_matrix = X_ids_leaves[i].astype(int)
        help_matrix = help_matrix.astype(float)
        help_matrix[help_matrix == 1] = weights_true
        weights_leaves.append(help_matrix)

    # 3. Calculate weight dependent on leave for bootstrap sample
    X_test_id_leaves = tree.apply(X_test)

    weights = []
    for i in range(len(X_test_id_leaves)):
        X_id = X_test_id_leaves[i]
        index = np.where(leaf_nodes == X_id)[0][0]  # Calculate index of test

        # Calculate mean through sum(weights * y_train)
        weights.append(weights_leaves[index])

    # Check if bootstrap true, otherwise continue
    if not bootstrap:
        final_weights = weights
    else:
        final_weights = []
        for i in range(len(X_test_id_leaves)):  # Itereate through X_test
            # Array with unique index values of X_train_base
            total_length = sampled_indices[weights[i] > 0].shape[0]
            base_indexes = np.unique(sampled_indices[weights[i] > 0])
            final_weight = np.zeros(X_train_base.shape[0])
            for index in range(len(base_indexes)):
                bootstrapped_times = sample_counts_dict[base_indexes[index]]
                final_weight[base_indexes[index]] = bootstrapped_times * (
                    1 / total_length
                )

            final_weights.append(final_weight)

    return np.array(final_weights)



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
        array: 3-dim-array with weights of rf
    """
    weights_rf = []
    for tree in rf.estimators_:
        weights_rf.append(
            calc_weights_tree(tree, X_train_base, X_test, bootstrap, max_samples)
        )
    return np.array(weights_rf)