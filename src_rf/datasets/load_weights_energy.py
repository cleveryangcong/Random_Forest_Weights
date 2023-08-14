import os
from scipy.sparse import load_npz
from scipy.sparse import vstack
import pandas as pd
from sklearn.model_selection import train_test_split
def load_weights_energy():
    def load_sparse_matrices_from_dir(dir_path):
        """Load all sparse matrices from a directory."""
        tree_files = sorted([os.path.join(dir_path, file) for file in os.listdir(dir_path)])
        return [load_npz(file) for file in tree_files]

    def load_all_rf_weights(base_dir, num_batches, num_trees):
        """Load and combine all rf weights."""
        all_rf_weights = [[] for _ in range(num_trees)]

        for batch_idx in range(num_batches):
            batch_dir = os.path.join(base_dir, f"batch_{batch_idx}")
            batch_weights = load_sparse_matrices_from_dir(batch_dir)

            for tree_idx, weights in enumerate(batch_weights):
                all_rf_weights[tree_idx].append(weights)

        # Now concatenate all batches for each tree
        for tree_idx in range(num_trees):
            all_rf_weights[tree_idx] = vstack(all_rf_weights[tree_idx])

        return all_rf_weights
    
    
    df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv"
                 , index_col = 'datetime', parse_dates=True)
    
    X = df.drop('total_energy_usage', axis = 1)
    y = df['total_energy_usage']
    # Before the Usage section:
    batch_size = 500  # This was the batch size you've defined earlier
    df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv", index_col="datetime")
    X = df.drop('total_energy_usage', axis=1).values
    _, X_test, _, _ = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
    num_samples = X_test.shape[0]

    # Usage
    base_dir = "/Data/Delong_BA_Data/rf_weights/"
    num_batches = (num_samples + batch_size - 1) // batch_size  # ceiling division to find number of batches
    # num_batches = 2
    num_trees = 300  # based on your RandomForestRegressor setup

    rf_weights_loaded = load_all_rf_weights(base_dir, num_batches, num_trees)
    return rf_weights_loaded