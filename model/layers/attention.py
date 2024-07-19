import tensorflow as tf
from tensorflow.python.ops import math_ops

class BahdanauAttention(tf.Keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
    
    def call(self, query, keys, mask):
        query_weights = tf.expand_dims(self.W1(query), 2)
        keys_weights = tf.expand_dims(self.W2(keys), 1)
        score = tf.reduce_sum(tf.nn.tanh(query_weights + keys_weights), -1)
        padding_mask = tf.expand_dims(math_ops.logical_not(mask), 1)
        score -= 1e9 * math_ops.cast(padding_mask, dtype=score.dtype)
        attention_scores = tf.expand_dims(tf.nn.softmax(score, axis=2), -1)
        context = tf.reduce_sum(attention_scores * tf.expand_dims(keys, axis=1), axis=2)
        return context, attention_scores
