'''
Run calc_weights function separately for different chunks of the test dataset through a for loop -> Avoid memory error
Parallize calculation of weights in for loop
Save calculated weights, in correct order
'''


# Basics
import pandas as pd
import numpy as np
import gc  # garbage collector

# Model
from sklearn.ensemble import RandomForestRegressor

# Helpful:
from sklearn.model_selection import train_test_split

# Path setup
import sys
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.append("/home/dchen/Random_Forest_Weights/")
# my functions:
from src_rf.methods.calc_mean import *
from src_rf.methods.calc_weights import *
from src_rf.methods.calc_dist import *


def compute_rf_weights(args):
    rf, X_train, batch, bootstrap, max_samples = args
    return calc_weights_rf(rf, X_train, batch, bootstrap, max_samples)


if __name__ == "__main__":
    # 1. Load Data:
    df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv", index_col="datetime")
    df.index = pd.to_datetime(df.index)

    # 2. Train Test Split:
    X = df.drop('total_energy_usage', axis=1).values
    y = df['total_energy_usage'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

    # 3. Random Forest:
    # 3.1 Parameters for Weight_Calculation:
    bootstrap = False
    max_samples = 0.8
    # 3.2 Parameters for RF

    # 3.3 Model Training
    rf = RandomForestRegressor(
        bootstrap=bootstrap, max_samples=max_samples, verbose=0, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 4. Parallel Processing:
    num_samples = X_test.shape[0]
    batch_size = 25
    file_path = "/Data/Delong_BA_Data/rf_weights/energy_weights.npy"
    # Define the number of processes you want to run in parallel
    max_workers = 1
    # Split the data into batches
    batches = [(X_test[i:i + batch_size], i, i + batch_size) for i in range(0, num_samples, batch_size)]

    # Use ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor(max_workers=1) as executor:  # Set max_workers to 2
        futures = [executor.submit(compute_rf_weights, (rf, X_train, batch, bootstrap, max_samples)) for batch, _, _ in batches]

        for future in futures:  # Iterate over futures in the order they were submitted
            rf_weights = future.result()

            if os.path.exists(file_path):
                existing_data = np.load(file_path)

                new_shape = (existing_data.shape[0], existing_data.shape[1] + rf_weights.shape[1], existing_data.shape[2])
                combined_data = np.zeros(new_shape, dtype=existing_data.dtype)

                combined_data[:, :existing_data.shape[1], :] = existing_data
                combined_data[:, existing_data.shape[1]:, :] = rf_weights

                np.save(file_path, combined_data)

                del existing_data  # Clear memory
            else:
                np.save(file_path, rf_weights)


            del rf_weights  # Explicitly delete to help with memory management
            gc.collect()  # Call garbage collector to free up memory

        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        