# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, time_steps):
    """Converts the time series data into sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        # Sequence (Input X): data[i] up to data[i + time_steps - 1]
        X.append(data[i:(i + time_steps), 0])
        # Target (Output y): data[i + time_steps]
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def load_and_prepare_data(file_path, time_steps=30, test_size=0.2):
    """Loads data, scales it, creates sequences, and splits into train/test sets."""
    
    # Check if the file exists (important for local testing)
    try:
        # Assuming the data file is a simple CSV without special indexing needs for now
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, None, None, None, None

    # Assuming the first column (index 0) is the target column based on your file name 'multivariate_timeseries.csv'
    # If your target column is 'Close' or something else, you MUST change df.iloc[:, 0:1]
    data = df.iloc[:, 0:1].values 
    
    # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, time_steps)
    
    # Reshape X for LSTM input: (samples, time_steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data into training and testing sets
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Data prepared successfully. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler
