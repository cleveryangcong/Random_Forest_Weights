# Dataset
from sklearn.datasets import load_diabetes

# Basics
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt

# Model
from sklearn.ensemble import RandomForestRegressor

# Helpful:
from sklearn.model_selection import train_test_split

# Path setup
import sys
import os

sys.path.insert(0, "/home/dchen/Random_Forest_Weights/")
# my functions:
from src.methods.calc_mean import *
from src.methods.calc_weights import *
from src.methods.calc_dist import *

def main():
    #1. Load Data
    df = pd.read_csv("datasets/energy_data_hourly.csv", index_col="datetime")
    df.index = pd.to_datetime(df.index)

    # Create the 'weekday' column
    df["weekday"] = df.index.day_name()
    # Create the 'time' column
    df["time"] = df.index.time
    df["weekday"] = df["weekday"].astype("category")
    df = pd.get_dummies(df, columns=["weekday"], prefix="", prefix_sep="")
    df["time"] = df["time"].apply(lambda t: t.hour * 60 + t.minute)


    #2. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, 1:].to_numpy(),
        df["total_energy_usage"].to_numpy(),
        test_size=0.2,
        shuffle=False,
    )


    #3. Random Forest
    bootstrap = True
    max_sample = 0.7
    rf = RandomForestRegressor(
        bootstrap=bootstrap, max_samples=max_sample, verbose=0, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_weights = calc_weights_rf(rf, X_train, X_test, bootstrap, max_sample)
    np.save("/home/dchen/Random_Forest_Weights/data/rf_weights/energy_data/rf_weights_True_0_7.npy", rf_weights)

if __name__ == "__main__":
    # Call the main function
    main()