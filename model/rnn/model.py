import tensorflow as tf

from tensorflow.keras.layers import LSTM

class RNN(tf.keras.Model):
    def __init__(self, units, embedding_size, vocab_size, input_length):
        super(RNN, self).__init__()
        self.input_length = input_length
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            input_length=input_length,
        )
        self.lstm = LSTM(units, embedding_size)
        self.classification_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_shape=(units,), activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

    def call(self, sentence):
        batch_size = tf.shape(sentence)[0]
        # Hidden state and Context state initialization
        pre_layer = tf.stack(
            tf.zeros([batch_size, self.units]),
            tf.zeros([batch_size, self.units])
        )
        # Embedding layer
        embedded_sentence = self.embedding(sentence)
        # LSTM + prev(Hidden state, Context state) -> curr(Hidden state, Context state)
        for i in range(self.input_length):
            # 1st : -> batch_size
            # i -> word position
            # 2nd : -> embedding
            # (batch_size, embedding_size)
            word = embedded_sentence[:, i, :]
            pre_layer = self.lstm(pre_layer, word)
        
        h, _ = tf.unstack(pre_layer)
        return self.classification_model(h)
