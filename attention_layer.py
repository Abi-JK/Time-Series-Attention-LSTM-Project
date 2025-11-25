# attention_layer.py

import tensorflow as tf
from tensorflow.keras.layers import Layer

class AttentionLayer(Layer):
    """Custom Attention Mechanism for LSTM output."""
    def _init_(self, **kwargs):
        super(AttentionLayer, self)._init_(**kwargs)

    def build(self, input_shape):
        # Weight matrix W for the context vector
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='W')
        # Bias vector b
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True,
                                 name='b')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 1. Calculate the attention score (e)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        # 2. Apply softmax to get attention weights (a)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # 3. Apply the weights to the original inputs (context vector)
        context_vector = attention_weights * inputs
        
        # 4. Sum across time steps to get the final context vector
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
    
    def get_config(self)(self):
        config = super(AttentionLayer, self).get_config()
        return config
