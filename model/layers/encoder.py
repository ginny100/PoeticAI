import tensorflow as tf

from tensorflow.keras.layers import Layer, Embedding, GRU # type: ignore

class Encoder(Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_lang_embedding = Embedding(input_vocab_size, embedding_dim)
        self.gru = GRU(self.enc_units, 
                       return_sequences=True, 
                       return_state=True, 
                       recurrent_initializer='glorot_uniform')
    
    def call(self, word_indices):
        word_embeddings = self.input_lang_embedding(word_indices)
        whole_sequence_output, final_state = self.gru(word_embeddings)
        return whole_sequence_output, final_state
