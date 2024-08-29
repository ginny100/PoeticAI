import tensorflow as tf

from tensorflow.keras.layers import Layer, Embedding, LSTM # type: ignore

class Encoder(Layer):
    def __init__(self, input_vocab_size, latent_dim):
        super(Encoder, self).__init__()
        # print("Encoder - latent_dim", latent_dim) # 50
        self.encoder_emb = Embedding(input_vocab_size, latent_dim, trainable=True) # Encoder
        self.encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True) # LSTM 1 
        self.encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True) # LSTM 2 
        self.encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True) # LSTM 3

    @tf.function
    def call(self, inputs):
        encoder_emb = self.encoder_emb(inputs)
        encoder_output1, _, _ = self.encoder_lstm1(encoder_emb)
        encoder_output2, _, _ = self.encoder_lstm2(encoder_output1)
        encoder_outputs, state_h, state_c = self.encoder_lstm3(encoder_output2) 
        # print("Encoder - state_h", state_h) # (None, 50)
        # print("Encoder - state_c", state_c) # (None, 50)
        return encoder_outputs, state_h, state_c
