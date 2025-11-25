# model.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
# .attention_layer 
from .attention_layer import AttentionLayer 

def create_attention_lstm_model(input_shape):
    """Creates the Attention-based LSTM model."""
    
    # Input shape is (time_steps, features)
    inputs = Input(shape=input_shape)
    
    # LSTM layer must return sequences for attention layer
    lstm_out = LSTM(units=64, return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Apply Attention Layer
    context_vector, attention_weights = AttentionLayer(name='attention_layer')(lstm_out)
    
    # Output layer
    outputs = Dense(1)(context_vector)
    
    # Model returns both prediction and weights (for visualization)
    model = Model(inputs=inputs, outputs=[outputs, attention_weights])
    model.compile(optimizer='adam', loss='mse')
    
    return model

def create_baseline_lstm_model(input_shape):
    """Creates a standard (baseline) LSTM model for comparison."""
    # This model acts as a baseline against the attention model.
    inputs = Input(shape=input_shape)
    # return_sequences=False as we only need the output of the last time step
    lstm_out = LSTM(units=64, return_sequences=False)(inputs) 
    lstm_out = Dropout(0.2)(lstm_out)
    outputs = Dense(1)(lstm_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model

# Note: ARIMA logic will be implemented separately in baseline_models.py
