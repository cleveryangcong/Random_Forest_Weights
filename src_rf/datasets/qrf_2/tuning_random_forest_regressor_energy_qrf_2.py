"""
0. Tuning
--> Should be the first step in a new dataset
Tuning of hyperparameters of a normal regression random forest from sklearn. Idea: Hyperparameters are similar to the ones that would be used for a quantile random forest
"""

# Path setup
import sys
import os

sys.path.append("/home/dchen/Random_Forest_Weights/")

# Basics:
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Helpful:
from sklearn.model_selection import train_test_split

# Pipeline and ColumnsTransformer:
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
# models:
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor

# my functions:
from src_rf.methods.calc_mean import *
from src_rf.methods.calc_weights import *
from src_rf.methods.calc_dist import *


df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv"
                 , index_col = 'datetime', parse_dates=True)

# Data manipulation to purely time series:
df.drop(['residual_energy_usage', 'pump_storage'], inplace = True, axis =  1)
# Extract the year from the index
df['Year'] = df.index.year

year_dummies = pd.get_dummies(df['Year'], prefix='Year')
month_dummies = pd.get_dummies(df['month'], prefix='Month')
hour_dummies = pd.get_dummies(df['hour'], prefix='Hour')

# Drop the original columns and join with dummy variables
df = df.drop(['Year', 'month', 'hour'], axis=1)
df = df.join([year_dummies, month_dummies, hour_dummies])

df['Count'] = range(0, df.shape[0])

X = df.drop('total_energy_usage', axis = 1)
y = df['total_energy_usage']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3 ,shuffle=False, random_state=42)

# Define the hyperparameters and their possible values
param_dist = {
    'n_estimators': [100, 300, 500, 700],
    'max_depth': [None,5, 10, 30, 50],
    'min_samples_split': [2,5,7, 10],
    'min_samples_leaf': [5, 10, 20],
    'bootstrap': [True],
    'max_samples': [None, 0.9, 0.7, 0.5, 0.3]
    
}


rf = RandomForestRegressor(random_state=42, n_jobs = -2)
tscv = TimeSeriesSplit(n_splits=5)

random_search = RandomizedSearchCV(
    rf, 
    param_distributions=param_dist, 
    n_iter=100, 
    cv=tscv, 
    verbose=1, 
    scoring='neg_mean_squared_error', # or any other appropriate scoring metric
    random_state=42
)


random_search.fit(X_train, y_train)

rf_cv_results = pd.DataFrame(random_search.cv_results_)

save_path = "/Data/Delong_BA_Data/rf_weights/qrf_2/rf_cv_results_1.csv"
rf_cv_results.to_csv(save_path, index=False)






