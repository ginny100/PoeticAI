import tensorflow as tf

from tensorflow.keras.layers import Layer, Input, Embedding, LSTM # type: ignore

class Decoder(Layer):
    def __init__(self, target_vocab_size, latent_dim):
        super(Decoder, self).__init__()
        self.decoder_inputs = Input(shape=(None,))
        self.decoder_emb = Embedding(target_vocab_size, latent_dim, trainable=True) # Decoder
        self.decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) # LSTM using encoder_states as initial state

    @tf.function
    def call(self, state_h, state_c):
        decoder_emb = self.decoder_emb(self.decoder_inputs)
        print("Decoder - decoder_emb", decoder_emb.shape) #
        decoder_outputs, decoder_fwd_state, decoder_back_state = self.decoder_lstm(decoder_emb, initial_state=[state_h, state_c])
        print("Decoder - decoder_outputs", decoder_outputs.shape) #
        print("Decoder - decoder_fwd_state", decoder_fwd_state.shape) # 
        print("Decoder - decoder_back_state", decoder_back_state.shape) # 
        return decoder_outputs, decoder_fwd_state, decoder_back_state