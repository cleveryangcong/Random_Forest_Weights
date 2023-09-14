import numpy as np
def filter_specific_weight_rf(rf_weights, test_data_index, batch_size):
    """
    Extracts and averages weights for a specific test data point from the random forest weights.

    Args:
    - rf_weights (list): A list of sparse matrices, each matrix having shape (num_X_test, num_leaves).
    - test_data_index (int): The index of the test data point for which the weights are to be extracted.
    - batch_size (int): The number of samples in each batch.

    Returns:
    - numpy.ndarray: An array of shape (num_leaves,) with the averaged weights.
    """
    
    # Calculate the local index within the batch for the test_data_index
    local_index_in_batch = test_data_index % batch_size

    # Extract the specific row (weights for local_index_in_batch) from each sparse matrix and convert to dense
    extracted_weights = [matrix[local_index_in_batch, :].toarray().squeeze() for matrix in rf_weights]
    
    # Stack them vertically to shape (num_trees, num_leaves)
    stacked_weights = np.vstack(extracted_weights)
    
    # Average over the number of trees (axis=0)
    averaged_weights = np.mean(stacked_weights, axis=0)
    
    return averaged_weights