'''
1. Calculate weights in batches
--> Should be run after deciding on the hyperparameters of the quantile random forest
Run calc_weights function separately for different chunks of the test dataset through a for loop -> Avoid memory error
Parallize calculation of weights in for loop
Save calculated weights, in correct order

Flexible -> Add flexible batch setting. As sometimes the program has problems with memory. Hence starting from a specific batch makes it easier to adjust.
'''

import sys
import os
import pandas as pd
import numpy as np
import gc  # garbage collector
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import save_npz, load_npz

sys.path.append("/home/dchen/Random_Forest_Weights/")
# my functions:
from src_rf.methods.calc_mean import *
from src_rf.methods.calc_weights import *
from src_rf.methods.calc_dist import *

def compute_rf_weights(args):
    rf, X_train, batch, bootstrap, max_samples = args
    return calc_weights_rf(rf, X_train, batch, bootstrap, max_samples)

def save_list_of_sparse_matrices(rf_weights, dir_path):
    """Save each sparse matrix in the list to a separate file within the specified directory."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for idx, weights in enumerate(rf_weights):
        save_npz(os.path.join(dir_path, f"tree_{idx}.npz"), weights)

def load_and_concat_sparse_matrices_from_dir(dir_path, num_trees):
    """Load and concatenate sparse matrices from a directory."""
    combined_data_list = []

    for idx in range(num_trees):
        # For each tree, load all the sparse matrices and concatenate them
        tree_files = sorted([os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.startswith(f"tree_{idx}_")])
        matrices = [load_npz(file) for file in tree_files]
        combined_data_list.append(sp.vstack(matrices))

    return combined_data_list

if __name__ == "__main__":
    # 1. Load Data:
    df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv", index_col="datetime")
    df.index = pd.to_datetime(df.index)
    df.drop(['residual_energy_usage', 'pump_storage'], inplace = True, axis =  1)
    # Extract the year from the index
    df['Year'] = df.index.year
    # 1. Extract weekday name
    df['weekday'] = df.index.day_name()

    # 2. Ordinal encode 'hour', 'weekday', 'month', and 'Year'
    # (In this case, 'hour', 'month', and 'Year' are already ordinal, so just encoding 'weekday')
    weekday_ordering = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df['weekday'] = df['weekday'].astype(pd.CategoricalDtype(categories=weekday_ordering, ordered=True))
    df['weekday'] = df['weekday'].cat.codes

    # No need to change the 'Year' column as you want it in ordinal form

    # 3. Add a count variable
    df['Count'] = range(df.shape[0])

    # Drop unnecessary columns
    columns_to_drop = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # 2. Train Test Split:
    X = df.drop('total_energy_usage', axis=1).values
    y = df['total_energy_usage'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

    # 3. Random Forest:
    # 3.1 Parameters for Weight_Calculation:
    bootstrap = True
    max_samples = 0.3
    # 3.2 Parameters for RF
    n_estimators = 300
    min_samples_split = 5
    min_samples_leaf = 10
    max_depth = 5

    # 3.3 Model Training
    rf = RandomForestRegressor(
        bootstrap=bootstrap, max_samples=max_samples,n_estimators = n_estimators, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_depth = max_depth, verbose=0, n_jobs=-1, random_state = 42
    )
    rf.fit(X_train, y_train)

    # 4. Parallel Processing:
    num_samples = X_test.shape[0]
    batch_size = 500
    dir_name = 'qrf_3'
    base_dir = f"/Data/Delong_BA_Data/rf_weights/{dir_name}/"
    # Check if directory exists, and if not, create it
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Start from batch_19
    start_batch = 19
    start_idx = start_batch * batch_size
    batches = [(X_test[i:i + batch_size], i, i + batch_size) for i in range(start_idx, num_samples, batch_size)]

    # Use ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor(max_workers=20) as executor:  # Set max_workers to 2
        futures = [executor.submit(compute_rf_weights, (rf, X_train, batch, bootstrap, max_samples)) for batch, _, _ in batches]

        for batch_idx, future in enumerate(futures, start=start_batch):  # Starting batch_idx from batch_19
            rf_weights = future.result()

            batch_dir = os.path.join(base_dir, f"batch_{batch_idx}")
            save_list_of_sparse_matrices(rf_weights, batch_dir)

            del rf_weights  # Explicitly delete to help with memory management
            gc.collect()  # Call garbage collector to free up memory