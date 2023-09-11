#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from joblib import Parallel, delayed

# Update the system path
sys.path.append("/home/dchen/Random_Forest_Weights/")
from src_rf.methods.calc_mean import *
from src_rf.methods.calc_weights import *
from src_rf.methods.calc_dist import *
from src_rf.datasets.load_weights_energy import * 

# Setup pandas display options
pd.set_option('display.max_columns', None)

def quantile_loss(y_true, y_pred, tau):
    return max(tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true))

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# Load Data and perform train-test split
df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv",
                 index_col='datetime', parse_dates=True)
X = df.drop('total_energy_usage', axis=1)
y = df['total_energy_usage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

# Load Random Forest Weights
rf_weights = load_weights_energy("/Data/Delong_BA_Data/rf_weights/")

# Calculate Quantiles
rf_dist = calc_dist_rf(rf_weights, y_train)

quantile_preds = np.zeros((len(y_test), 5))
for count, q in enumerate(quantiles):
    quantile_preds[:,count] = np.array(calc_quantile_rf(rf_dist,q, y_train))

path = '/Data/Delong_BA_Data/rf_weights/energy_quantile_preds.npy'
np.save(path, quantile_preds)







