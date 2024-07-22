import tensorflow as tf

from tensorflow.keras.layers import Layer, Embedding, LSTM # type: ignore

class Encoder(Layer):
    def __init__(self, encoder_input, input_vocab_size, latent_dim):
        super(Encoder, self).__init__()
        self.encoder_emb = Embedding(input_vocab_size, latent_dim, trainable=True)(encoder_input) # Encoder
        self.encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True) # LSTM 1 
        self.encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True) # LSTM 2 
        self.encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True) # LSTM 3

    def call(self):
        encoder_output1, _, _ = self.encoder_lstm1(self.encoder_emb)
        encoder_output2, _, _ = self.encoder_lstm2(encoder_output1)
        encoder_outputs, state_h, state_c = self.encoder_lstm3(encoder_output2) 
        return encoder_outputs, state_h, state_c
