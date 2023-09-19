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
from src_rf.datasets.qrf_3.load_weights_qrf_3 import * 
from src_rf.datasets.qrf_3.load_specific_weights_qrf_3 import * 
from src_rf.methods.filter_weights_rf import * 
from src_rf.methods.filter_specific_weights_rf import * 

# Setup pandas display options
pd.set_option('display.max_columns', None)

def quantile_loss(y_true, y_pred, tau):
    return max(tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true))

quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

# Load Data and perform train-test split
df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv",
                 index_col='datetime', parse_dates=True)

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

X = df.drop('total_energy_usage', axis=1)
y = df['total_energy_usage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

# Load Random Forest Weights by batches
dir_path = "/Data/Delong_BA_Data/rf_weights/qrf_3/"
batch_size = 500
iterations = (X_test.shape[0] // batch_size) + 1
quantile_preds = np.zeros((len(y_test), 5))
for i in range(iterations):
    # Load Random Forest Weights
    rf_weights = load_specific_weights_qrf_3(dir_path, i * 500)
    # Calculate Quantiles
    rf_dist = calc_dist_rf(rf_weights, y_train)
    for count, q in enumerate(quantiles):
        if i != (iterations - 1):
            quantile_preds[(i * 500):(i* 500) + 500,count] = np.array(calc_quantile_rf(rf_dist,q, y_train))
        else:
            quantile_preds[(i * 500):X_test.shape[0],count] = np.array(calc_quantile_rf(rf_dist,q, y_train))

path = '/Data/Delong_BA_Data/rf_weights/qrf_3/energy_quantile_preds_qrf_3.npy'
np.save(path, quantile_preds)