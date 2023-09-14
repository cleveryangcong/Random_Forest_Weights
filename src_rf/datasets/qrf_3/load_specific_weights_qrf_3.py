import os
from scipy.sparse import load_npz
from scipy.sparse import vstack
import pandas as pd
from sklearn.model_selection import train_test_split

def load_weights_qrf_3(base_dir, test_data_index):
    def load_sparse_matrices_from_dir(dir_path):
        """Load all sparse matrices from a directory."""
        tree_files = sorted([os.path.join(dir_path, file) for file in os.listdir(dir_path)])
        return [load_npz(file) for file in tree_files]

    def load_specific_rf_weights(base_dir, batch_idx, num_trees):
        """Load weights for a specific batch."""
        batch_dir = os.path.join(base_dir, f"batch_{batch_idx}")
        batch_weights = load_sparse_matrices_from_dir(batch_dir)
        return batch_weights
    
    
    df = pd.read_csv("/home/dchen/Random_Forest_Weights/src_rf/data/energy_data_hourly.csv"
                 , index_col = 'datetime', parse_dates=True)
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
    
    X = df.drop('total_energy_usage', axis = 1)
    y = df['total_energy_usage']
    
    # Before the Usage section:
    batch_size = 500  # This was the batch size you've defined earlier

    X = df.drop('total_energy_usage', axis=1).values
    _, X_test, _, _ = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

    # Determine which batch the test_data_index belongs to
    batch_idx_for_test_data = test_data_index // batch_size

    # Use the modified function to load only the specific batch's weight
    rf_weights_loaded = load_specific_rf_weights(base_dir, batch_idx_for_test_data, num_trees)
    return rf_weights_loaded
    
    
    
    
    
    
    
    
    
    
    
    
    