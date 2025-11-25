# main.py

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from model import create_attention_lstm_model, create_baseline_lstm_model
from data_preprocessing import load_and_prepare_data
from visualization_utils import plot_predictions, plot_attention_weights
from baseline_models import run_arima 

# --- Configuration ---
DATA_FILE = 'multivariate_timeseries.csv'
TIME_STEPS = 30
RESULTS_DIR = 'results'

# Ensure the results directory exists (Fix for Deliverable 3)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- 1. Load and Prepare Data ---
X_train, y_train, X_test, y_test, scaler = load_and_prepare_data(DATA_FILE, time_steps=TIME_STEPS)

if X_train is None:
    print("Exiting due to data loading error.")
    exit()

# Get the original scaled test targets (required for ARIMA evaluation)
y_test_descaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_train_descaled = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()


input_shape = (X_train.shape[1], X_train.shape[2])

# --- 2. Attention LSTM ---
att_lstm_model = create_attention_lstm_model(input_shape)
print("\n--- Training Attention LSTM ---")
att_lstm_model.fit(X_train, [y_train, np.zeros_like(y_train)], 
                   epochs=50, batch_size=32, verbose=0) 

predictions_att, attention_weights = att_lstm_model.predict(X_test)
predictions_att_descaled = scaler.inverse_transform(predictions_att).flatten()

# Metrics
rmse_att = np.sqrt(mean_squared_error(y_test_descaled, predictions_att_descaled))
mae_att = mean_absolute_error(y_test_descaled, predictions_att_descaled)
mape_att = mean_absolute_percentage_error(y_test_descaled, predictions_att_descaled)

# Visualizations (Fix for Deliverable 3)
plot_predictions(y_test_descaled, predictions_att_descaled, "Attention_LSTM", RESULTS_DIR)
plot_attention_weights(X_test, attention_weights, sample_index=5, path=RESULTS_DIR) 

# --- 3. Baseline LSTM ---
baseline_lstm_model = create_baseline_lstm_model(input_shape)
print("\n--- Training Baseline LSTM ---")
baseline_lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0) 

predictions_base = baseline_lstm_model.predict(X_test)
predictions_base_descaled = scaler.inverse_transform(predictions_base).flatten()

# Metrics
rmse_base = np.sqrt(mean_squared_error(y_test_descaled, predictions_base_descaled))
mae_base = mean_absolute_error(y_test_descaled, predictions_base_descaled)
mape_base = mean_absolute_percentage_error(y_test_descaled, predictions_base_descaled)

plot_predictions(y_test_descaled, predictions_base_descaled, "Baseline_LSTM", RESULTS_DIR)


# --- 4. ARIMA ---
print("\n--- Running ARIMA ---")
arima_results = run_arima(y_train_descaled, y_test_descaled) 
rmse_arima = arima_results['rmse']
mae_arima = arima_results['mae']
mape_arima = arima_results['mape']
plot_predictions(y_test_descaled, arima_results['predictions'], "ARIMA", RESULTS_DIR)


# --- 5. Final Comparison (Fix for Critical Deliverable 2) ---
print("\n--- Final Performance Metrics ---")
comparison_table = pd.DataFrame({
    'Model': ['Attention LSTM', 'Baseline LSTM', 'ARIMA'],
    'RMSE': [rmse_att, rmse_base, rmse_arima],
    'MAE': [mae_att, mae_base, mae_arima],
    'MAPE': [mape_att, mape_base, mape_arima]
})

print(comparison_table.to_markdown(index=False))
comparison_table.to_csv(os.path.join(RESULTS_DIR, 'comparison_metrics.csv'), index=False)
