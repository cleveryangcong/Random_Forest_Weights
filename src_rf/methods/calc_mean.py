from scipy.sparse import csr_matrix
import numpy as np

def calc_mean_tree(weights_tree, y_train):
    """
        Function that calculates the mean prediction of tree based on the weights of a given X_test input and the y_train sample
    Args:
        weights_tree(Union[csr_matrix, np.ndarray]): Sparse matrix or numpy array with the weights for the training sample.
        y_train(array): Array with y values used for training of the specific tree
    Returns:
        array: array of mean values
    """
    result = weights_tree.dot(y_train)
    
    if isinstance(result, csr_matrix):
        means = result.toarray().squeeze()
    else:
        # It's already a numpy array
        means = result.squeeze()

    return means

    return means
def calc_mean_rf(weights_rf, y_train):
    """
        Function that calculates the mean prediction of rf based on the weights of a given X_test input and the y_train sample
    Args:
        weights_rf(list[csr_matrix]): List of sparse matrices representing weights for each tree in the forest
        y_train(array): Array with y values used for training of the specific tree
    Returns:
        array: array of mean values
    """
    # Calculate the means for each tree using a list comprehension
    tree_means = [calc_mean_tree(tree_weights, y_train) for tree_weights in weights_rf]
    
    # Use numpy operations for efficient mean calculation across trees
    means = np.mean(tree_means, axis=0)
    
    return means
