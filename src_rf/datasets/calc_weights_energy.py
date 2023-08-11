'''
Run calc_weights function separately for different chunks of the test dataset through a for loop -> Avoid memory error
Parallize calculation of weights in for loop
Save calculated weights, in correct order
'''


# Basics
import pandas as pd
import numpy as np

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
    X = df.drop('total_energy_usage', axis = 1).values
    y = df['total_energy_usage'].values    
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state = 42)   
    
    # 3. Random Forest:
        # 3.1 Parameters for Weight_Calculation:
    bootstrap = False
    max_samples = 0.8
        # 3.2 Parameters for RF
    
        # 3.3 Model Training
    rf = RandomForestRegressor(
    bootstrap=bootstrap, max_samples = max_samples, verbose=0, n_jobs=-1
)
    rf.fit(X_train, y_train)    
    
    # 4. Parallel Processing:
    num_samples = X_test.shape[0]
    batch_size = 50
    file_path = "/home/dchen/Random_Forest_Weights/data/rf_weights/energy_weights.npy"
    # Define the number of processes you want to run in parallel
    max_workers = 2
    # Split the data into batches
    batches = [(X_test[i:i+batch_size], i, i+batch_size) for i in range(0, num_samples, batch_size)]

    # Use ProcessPoolExecutor to parallelize computation
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        args = [(rf, X_train, batch, bootstrap, max_samples) for batch, _, _ in batches]
        results = list(executor.map(compute_rf_weights, args))

    # Now, save the results batch-wise to ensure order
    for rf_weights in results:
        # Load existing data, concatenate and save back
        if os.path.exists(file_path):
            existing_data = np.load(file_path)
            combined_data = np.concatenate([existing_data, rf_weights], axis=0)
        else:
            combined_data = rf_weights

        np.save(file_path, combined_data)

        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        