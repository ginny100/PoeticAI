import tensorflow as tf

class LSTM(tf.keras.layers.Layer):
    def __init__(self, units, inp_shape):
        super(LSTM, self).__init__()
        self.units = units
        self.inp_shape = inp_shape
        self.W = self.add_weight(name="W", shape=(4, self.units, self.inp_shape))
        self.U = self.add_weight(name="U", shape=(4, self.units, self.units))
    
    def call(self, pre_layer, x):
        pre_h, pre_c = tf.unstack(pre_layer)
        # Input gate
        i_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[0])) + tf.matmul(pre_h, tf.transpose(self.U[0])))
        # Forget gate
        f_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[1])) + tf.matmul(pre_h, tf.transpose(self.U[1])))
        # Output gate
        o_t = tf.nn.sigmoid(tf.matmul(x, tf.transpose(self.W[2])) + tf.matmul(pre_h, tf.transpose(self.U[2])))
        # New cell memory
        n_c_t = tf.nn.tanh(tf.matmul(x, tf.transpose(self.W[3])) + tf.matmul(pre_h, tf.transpose(self.U[3])))
        # Current cell memory
        c = tf.multiply(f_t, pre_c) + tf.multiply(i_t, n_c_t)
        # Current hidden state
        h = tf.multiply(o_t, tf.nn.tanh(c))
        return tf.stack([h, c])
