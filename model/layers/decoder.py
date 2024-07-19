import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, GRU # type: ignore
from model.layers.attention import BahdanauAttention

class Decoder(Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.english_embedding = Embedding(output_vocab_size, embedding_dim)
        self.gru = GRU(dec_units, return_sequences=True, return_state=True)
        self.dense = tf.Keras.layers.Dense(output_vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def call(self, input_word_indices, encoder_keys, mask, state=None):
        embedding_ = self.english_embedding(input_word_indices)
        output, state = self.gru(embedding_, initial_state=state)
        context, _ = self.attention(query=output, keys=encoder_keys, mask = mask)
        concat = tf.concat([output, context], axis=-1)
        vocab_output = self.dense(concat)
        return vocab_output, state