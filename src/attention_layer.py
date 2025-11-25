

import tensorflow as tf
K = tf.keras.backend

#Custom Attention Layer Implementation 
class AdditiveAttentionLayer(tf.keras.layers.Layer):
    """
    Bahdanau-style Additive Attention Mechanism.
    Calculates context vector based on decoder state and encoder outputs (H_t) and last hidden state (S_t).
    """
    def __init__(self, **kwargs):
        super(AdditiveAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(feature_dim, feature_dim),
                                 initializer='glorot_uniform', name='W')
        self.U = self.add_weight(shape=(feature_dim, feature_dim),
                                 initializer='glorot_uniform', name='U')
        self.V = self.add_weight(shape=(feature_dim, 1),
                                 initializer='glorot_uniform', name='V')
        super(AdditiveAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        encoder_outputs, last_hidden_state = inputs
        last_hidden_state_expanded = K.expand_dims(last_hidden_state, 1)

        # 1. Alignment Score Calculation (Additive/Bahdanau)
        score = K.tanh(K.dot(encoder_outputs, self.W) + K.dot(last_hidden_state_expanded, self.U))
        
        # 2. Attention Weights (Softmax over time steps)
        attention_weights = K.softmax(K.dot(score, self.V), axis=1) 
        
        # 3. Context Vector
        context_vector = K.sum(attention_weights * encoder_outputs, axis=1)

        return [context_vector, attention_weights]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][2]), (input_shape[0][0], input_shape[0][1], 1)]

# Model Definition Functions

def create_lstm(input_shape: tuple, units: int = 64) -> tf.keras.Model:
    """Standard LSTM model without attention (Baseline DL Model)."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    lstm_out = tf.keras.layers.LSTM(units, activation='relu', return_sequences=False)(inputs)
    output = tf.keras.layers.Dense(1)(lstm_out)
    model = tf.keras.models.Model(inputs=inputs, outputs=output, name='Standard_LSTM')
    model.compile(optimizer='adam', loss='mse')
    return model

def create_attention_lstm(input_shape: tuple, units: int = 64) -> tf.keras.Model:
    """LSTM model incorporating the Additive Attention layer (Core Model)."""

    inputs = tf.keras.layers.Input(shape=input_shape)
    # The LSTM encoder must return sequences (H_t) AND the last state (S_t)
    lstm_out, state_h, state_c = tf.keras.layers.LSTM(units, activation='relu',
                                                      return_sequences=True,
                                                      return_state=True)(inputs)

    # Use the custom attention layer
    context_vector, attention_weights = AdditiveAttentionLayer()([lstm_out, state_h])

    # Concatenate context vector (C_t) with the last hidden state (S_t)
    merged_vector = tf.keras.layers.Concatenate(axis=-1)([state_h, context_vector])

    output = tf.keras.layers.Dense(1)(merged_vector)

    # Model outputs the prediction and the attention weights for analysis
    model = tf.keras.models.Model(inputs=inputs, outputs=[output, attention_weights], name='Attention_LSTM')
    model.compile(optimizer='adam', loss='mse')
    return model
