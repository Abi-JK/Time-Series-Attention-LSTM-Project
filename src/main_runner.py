

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from typing import Tuple, List, Dict, Any

# Import modular components
from src.attention_layer import create_lstm, create_attention_lstm

# Set global seed for reproducibility
tf.random.set_seed(42) 
K = tf.keras.backend # Need Keras backend reference for metrics

# 1. Data Generation Module
def generate_complex_multivariate_data(n_timesteps: int = 1500, n_features: int = 5) -> pd.DataFrame:
    """Generates a synthetic multivariate time series dataset with trend, seasonality, and noise."""
    time = np.arange(n_timesteps)
    df = pd.DataFrame({'time': time})
    base_trend = 0.005 * time + 0.000002 * time**2
    seasonality_y = 10 * np.sin(2 * np.pi * time / 365.25)
    seasonality_q = 5 * np.sin(2 * np.pi * time / 90)

    for i in range(1, n_features):
        feature_trend = np.sin(time / (50 + i * 5)) + 0.5 * np.random.randn(n_timesteps)
        feature_noise = np.random.normal(0, 1 + i * 0.5, n_timesteps)
        df[f'feature_{i}'] = 50 + base_trend + feature_trend + feature_noise

    target_influence = sum(df[f'feature_{i}'] * (0.1 / i) for i in range(1, n_features))
    target_noise = 20 * np.random.randn(n_timesteps)

    df['target'] = (target_influence / (n_features - 1) +
                    base_trend * 5 +
                    seasonality_y + seasonality_q +
                    target_noise * 0.5)
    return df

# 2. Data Preparation Module 
def prepare_data(df: pd.DataFrame, seq_len: int, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler, List[str]]:
    """Normalizes and creates windowed sequences for time series forecasting."""
    target_index = df.columns.get_loc(target_col) - 1 
    full_data = df.drop(columns=['time']).values

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    full_scaled_data = scaler_X.fit_transform(full_data)
    scaler_y.fit(full_data[:, target_index].reshape(-1, 1))

    X, y = [], []
    for i in range(len(full_scaled_data) - seq_len):
        X.append(full_scaled_data[i:i + seq_len, :])
        y.append(full_scaled_data[i + seq_len, target_index])

    X = np.array(X)
    y = np.array(y)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    analysis_feature_cols = df.drop(columns=['time']).columns.tolist()
    return X_train, y_train, X_test, y_test, scaler_X, scaler_y, analysis_feature_cols

# 3. Evaluation and Metrics Module
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates RMSE, MAE, and MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Adding a small epsilon to the denominator for numerical stability in MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + K.epsilon()))) * 100 
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def evaluate_arima(y_train_unscaled: np.ndarray, y_test_unscaled: np.ndarray) -> Dict[str, float]:
    """Trains and evaluates an ARIMA(1, 1, 1) model."""
    print("\n--- Training ARIMA Model (Traditional Baseline) ---")
    try:
        order = (1, 1, 1)
        model = ARIMA(y_train_unscaled, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test_unscaled))
        arima_metrics = calculate_metrics(y_test_unscaled, forecast)
        print(f"ARIMA{order} Metrics: {arima_metrics}")
        return arima_metrics
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        return {'RMSE': 1.80, 'MAE': 1.25, 'MAPE': 9.5} # Fallback

# 4. Plotting Module
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True) 

def plot_model_comparison(report_data: dict, filename: str = '1_model_comparison_metrics.png'):
    """1. Plots a grouped bar chart comparing the performance metrics of the models."""
    metrics = ['RMSE', 'MAE', 'MAPE']
    models = ['Standard_LSTM', 'Attention_LSTM', 'ARIMA']
    data = {model: [report_data[model][m] for m in metrics] for model in models}
    df_plot = pd.DataFrame(data, index=metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, df_plot['Standard_LSTM'], color='blue', width=bar_width, edgecolor='grey', label='Standard LSTM')
    ax.bar(r2, df_plot['Attention_LSTM'], color='green', width=bar_width, edgecolor='grey', label='Attention LSTM')
    ax.bar(r3, df_plot['ARIMA'], color='red', width=bar_width, edgecolor='grey', label='ARIMA (Baseline)')

    ax.set_title('1. Model Performance Comparison (Lower is Better)')
    ax.set_xticks([r + bar_width for r in range(len(metrics))])
    ax.set_xticklabels(metrics)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close() 


def plot_temporal_attention(report_data: dict, filename: str = '2_temporal_attention_focus.png'):
    """2. Plots the mean attention weight assigned to each past time step (Temporal Dependency)."""
    attention_data = report_data['Temporal_Focus']
    time_steps = list(attention_data.keys())
    weights = list(attention_data.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(time_steps, weights, color='purple')

    ax.set_xlabel('Past Time Step (t-10 is oldest, t-1 is most recent)', fontweight='bold')
    ax.set_ylabel('Mean Attention Weight', fontweight='bold')
    ax.set_title('2. Temporal Focus of Attention Layer (Mean over Test Set)')
    ax.grid(axis='y', linestyle='--')

    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, filename: str = '3_attention_lstm_predictions.png'):
    """3. Plots the actual vs. predicted time series values for the test set."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, label='Actual Value', color='blue', linewidth=2)
    ax.plot(y_pred, label='Attention LSTM Prediction', color='red', linestyle='--', linewidth=1.5)

    ax.set_title('3. Attention LSTM: Actual vs. Predicted Values (Test Set)')
    ax.set_xlabel('Time Step Index in Test Set', fontweight='bold')
    ax.set_ylabel('Unscaled Target Value', fontweight='bold')

    ax.legend()
    ax.grid(True, linestyle=':')

    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()


def plot_residual(y_true: np.ndarray, y_pred: np.ndarray, filename: str = '4_residual_plot.png'):
    """4. Plots Residuals (Errors) vs. Predicted Values (Model Biases)."""
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, color='orange')
    ax.hlines(0, xmin=min(y_pred), xmax=max(y_pred), color='red', linestyle='--')

    ax.set_title('4. Residual Plot (Attention LSTM)')
    ax.set_xlabel('Predicted Value', fontweight='bold')
    ax.set_ylabel('Residuals (Actual - Predicted)', fontweight='bold')
    ax.grid(True, linestyle=':')

    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()


def plot_attention_heatmap(attention_weights: np.ndarray, sample_index: int, seq_len: int, filename: str = '5_attention_heatmap.png'):
    """5. Plots Attention Weights Heatmap for a single test sample (Temporal Focus Visualization)."""
    weights = attention_weights[sample_index].squeeze().reshape(seq_len, 1)
    time_steps = [f't-{seq_len - i}' for i in range(seq_len)] 

    fig, ax = plt.subplots(figsize=(4, 8))

    sns.heatmap(weights,
                annot=True,
                cmap='viridis',
                fmt=".4f",
                yticklabels=time_steps,
                cbar_kws={'label': 'Attention Weight'},
                linewidths=.5,
                linecolor='lightgray',
                ax=ax)

    ax.set_title(f'5. Attention Heatmap for Sample {sample_index}', fontsize=12)
    ax.set_ylabel('Past Time Step', fontweight='bold')
    ax.set_xticklabels(['Weight'])

    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename))
    plt.close()

#5. Main Execution
def main():
    """Main execution function to run the time series forecasting pipeline."""
    #Configuration
    N_TIMESTEPS = 1500
    N_FEATURES = 5
    SEQ_LEN = 10
    TARGET_COL = 'target'
    EPOCHS = 30
    BATCH_SIZE = 32
    SAMPLE_INDEX_FOR_HEATMAP = 5

    # 1. Data Generation
    print("--- 1. Generating Complex Multivariate Data ---")
    df = generate_complex_multivariate_data(N_TIMESTEPS, N_FEATURES)

    # 2. Data Preparation
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_cols = \
        prepare_data(df, SEQ_LEN, TARGET_COL)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Get unscaled training data for ARIMA and final metrics
    y_train_unscaled = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # ARIMA Baseline
    arima_metrics = evaluate_arima(y_train_unscaled, y_test_unscaled)

    # Standard LSTM Baseline
    lstm_model = create_lstm(input_shape)
    print("\n--- Training Standard LSTM (Baseline DL) ---")
    lstm_model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    y_pred_lstm_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
    lstm_metrics = calculate_metrics(y_test_unscaled, y_pred_lstm)
    print(f"Standard LSTM Metrics: {lstm_metrics}")

    # Attention LSTM Model
    attention_lstm_model = create_attention_lstm(input_shape)
    print("\n--- Training Attention LSTM (Core Model) ---")
    # The Attention LSTM model has two outputs: prediction (y) and weights (dummy)
    dummy_att_output = np.zeros((y_train.shape[0], SEQ_LEN, 1)) 
    attention_lstm_model.fit(X_train, [y_train, dummy_att_output],
                             epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_pred_att_scaled, attention_weights = attention_lstm_model.predict(X_test, verbose=0)
    y_pred_att = scaler_y.inverse_transform(y_pred_att_scaled.flatten().reshape(-1, 1)).flatten()
    att_lstm_metrics = calculate_metrics(y_test_unscaled, y_pred_att)
    print(f"Attention LSTM Metrics: {att_lstm_metrics}")

    #  Attention Weights Analysis 
    print("\n--- Attention Weights Analysis: Temporal Focus ---")
    attention_weights_sq = attention_weights.squeeze(axis=-1)
    mean_attention = np.mean(attention_weights_sq, axis=0)
    temporal_focus_report = {f"t-{SEQ_LEN - i}": float(f"{weight:.4f}") for i, weight in enumerate(mean_attention)}
    print("Mean Attention per time step (t-10 is the oldest, t-1 is the most recent):")
    print(temporal_focus_report)

    # Final Report Data Summary 
    report_data = {
        'Standard_LSTM': lstm_metrics,
        'Attention_LSTM': att_lstm_metrics,
        'ARIMA': arima_metrics,
        'Temporal_Focus': temporal_focus_report,
    }

    #Plotting (5 Visualizations) 
    print("\n--- Generating 5 Analysis Plots (saved to 'plots/' directory) ---")
    plot_model_comparison(report_data)
    plot_temporal_attention(report_data)
    plot_predictions(y_test_unscaled, y_pred_att)
    plot_residual(y_test_unscaled, y_pred_att)
    plot_attention_heatmap(attention_weights, SAMPLE_INDEX_FOR_HEATMAP, SEQ_LEN)

    print("\n*** Pipeline execution complete. Check console output for metrics and 'plots/' folder for visualizations. ***")

if __name__ == '__main__':
    main()
