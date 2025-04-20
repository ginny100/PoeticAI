import tensorflow as tf
from model.layers.lstm_cell import LSTM

class Bidirectional(tf.keras.Model):
    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(Bidirectional, self).__init__()
        self.input_length = input_length
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length,
        )
        self.lstm_fw = LSTM(units, embedding_size)  # Forward LSTM
        self.lstm_bw = LSTM(units, embedding_size)  # Backward LSTM
        self.classification_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(2 * units,), activation='relu'),
            tf.keras.layers.Dense(vocab_size, activation='softmax')
        ])

    def call(self, sentence):
        batch_size = tf.shape(sentence)[0]
        
        # Initial Hidden states and Context states for forward and backward LSTM
        pre_layer_fw = tf.stack([
            tf.zeros([batch_size, self.units]),
            tf.zeros([batch_size, self.units]) 
        ])
        pre_layer_bw = tf.stack([
            tf.zeros([batch_size, self.units]),
            tf.zeros([batch_size, self.units])
        ])
        
        # Embedding layer
        embedded_sentence = self.embedding(sentence)
        
        # Forward LSTM
        for i in range(self.input_length):
            word_fw = embedded_sentence[:, i, :]
            pre_layer_fw = self.lstm_fw(pre_layer_fw, word_fw)
        
        # Backward LSTM
        for i in range(self.input_length-1, -1, -1):
            word_bw = embedded_sentence[:, i, :]
            pre_layer_bw = self.lstm_bw(pre_layer_bw, word_bw)
        
        h_fw, _ = tf.unstack(pre_layer_fw)
        h_bw, _ = tf.unstack(pre_layer_bw)
        
        # Concatenate forward and backward hidden states
        h_concat = tf.concat([h_fw, h_bw], axis=-1)
        
        return self.classification_model(h_concat)
