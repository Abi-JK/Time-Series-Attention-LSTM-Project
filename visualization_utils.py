# visualization_utils.py

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_actual, y_predicted, model_name, path='results'):
    """Plots the actual time series values against the predicted values."""
    plt.figure(figsize=(12, 6))
    index = np.arange(len(y_actual))

    plt.plot(index, y_actual, label='Actual Values', color='blue')
    plt.plot(index, y_predicted, label='Predicted Values', color='red', linestyle='--')
    
    plt.title(f"{model_name} Prediction vs. Actual")
    plt.xlabel('Time Step (Test Period)')
    plt.ylabel('Value (Original Scale)')
    plt.legend()
    plt.grid(True)
    # Saves the plot to the results folder
    plt.savefig(f'{path}/{model_name}_Prediction.png')
    plt.close()
    # [attachment_0](attachment)

def plot_attention_weights(X_sequence, attention_weights, sample_index=0, path='results'):
    """Visualizes the attention weights for a specific input sequence."""
    
    # Select a single sequence (X) and its corresponding weights (A)
    sequence = X_sequence[sample_index, :, 0]
    weights = attention_weights[sample_index, :, 0]

    # Normalize weights for color mapping
    weights_normalized = (weights - weights.min()) / (weights.max() - weights.min())

    plt.figure(figsize=(10, 5))
    
    # Plot the input sequence, colored by the attention weights
    plt.bar(range(len(sequence)), sequence, color=plt.cm.viridis(weights_normalized), alpha=0.9)
    
    plt.title(f"Attention Weights Visualization (Sample {sample_index})")
    plt.xlabel('Time Step in Sequence (Lookback Period)')
    plt.ylabel('Input Value (Scaled)')
    
    # Color bar legend
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=weights.min(), vmax=weights.max()))
    sm.set_array([])
    plt.colorbar(sm, orientation='vertical', label='Attention Weight Magnitude')
    
    plt.savefig(f'{path}/Attention_Weights_Sample_{sample_index}.png')
    plt.close()
    #
