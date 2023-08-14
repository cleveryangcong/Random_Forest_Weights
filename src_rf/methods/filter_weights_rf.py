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

# models:
import statsmodels.api as sm

# my functions:
from src_rf.methods.calc_mean import *
from src_rf.methods.calc_weights import *
from src_rf.methods.calc_dist import *
from src_rf.datasets.load_weights_energy import * 

def filter_weight_rf(rf_weights, test_data_index):
    """
    Extracts and averages weights for a specific test data point from the random forest weights.

    Args:
    - rf_weights (list): A list of sparse matrices, each matrix having shape (num_X_test, num_leaves).
    - test_data_index (int): The index of the test data point for which the weights are to be extracted.

    Returns:
    - numpy.ndarray: An array of shape (num_leaves,) with the averaged weights.
    """
    
    # Extract the specific row (weights for test_data_index) from each sparse matrix and convert to dense
    extracted_weights = [matrix[test_data_index, :].toarray().squeeze() for matrix in rf_weights]
    
    # Stack them vertically to shape (num_trees, num_leaves)
    stacked_weights = np.vstack(extracted_weights)
    
    # Average over the number of trees (axis=0)
    averaged_weights = np.mean(stacked_weights, axis=0)
    
    return averaged_weights














