# baseline_models.py

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from data_preprocessing import load_and_prepare_data # Importing data preparation utility

def run_arima(y_train_descaled, y_test_descaled, order=(5,1,0)):
    """
    Trains and predicts using the ARIMA model (Autoregressive Integrated Moving Average).
    
    Args:
        y_train_descaled (np.array): Training data (original scale).
        y_test_descaled (np.array): Test data (original scale).
        order (tuple): ARIMA parameters (p, d, q).
        
    Returns:
        dict: Dictionary containing the calculated metrics.
    """
    # Combine train and test for easier forecasting context
    train_size = len(y_train_descaled)
    history = [x for x in y_train_descaled]
    predictions = []
    
    print("Running ARIMA model...")
    
    # ARIMA forecasting loop
    for t in range(len(y_test_descaled)):
        try:
            # Model training on history
            model = ARIMA(history, order=order)
            model_fit = model.fit()
            
            # Forecast one step ahead
            y_hat = model_fit.forecast(steps=1)[0]
            predictions.append(y_hat)
            
            # Add true value to history for next forecast step (Walk Forward Validation)
            history.append(y_test_descaled[t]) 
        except Exception as e:
            # Simple handling for common ARIMA convergence errors
            predictions.append(history[-1]) # Use last known value if error occurs
            history.append(y_test_descaled[t])
            print(f"ARIMA error: {e}. Using last value for forecast.")

    predictions = np.array(predictions).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_descaled, predictions))
    mae = mean_absolute_error(y_test_descaled, predictions)
    mape = mean_absolute_percentage_error(y_test_descaled, predictions)
    
    print(f"ARIMA Metrics calculated.")
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions
    }
