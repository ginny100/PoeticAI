import tensorflow as tf

from tensorflow.keras.layers import Layer, Embedding, LSTM # type: ignore

class Decoder(Layer):
    def __init__(self, target_vocab_size, latent_dim):
        super(Decoder, self).__init__()
        self.decoder_emb = Embedding(target_vocab_size, latent_dim, trainable=True) # Decoder
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) # LSTM using encoder_states as initial state

    @tf.function
    def call(self, inputs, state_h, state_c):
        decoder_emb = self.decoder_emb(inputs)
        decoder_outputs, decoder_fwd_state, decoder_back_state = self.decoder_lstm(decoder_emb, initial_state=[state_h, state_c])
        return decoder_outputs, decoder_fwd_state, decoder_back_state
    