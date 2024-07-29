import tensorflow as tf

from model.layer import Decoder
from model.layer import Encoder
from tensorflow.keras.layers import Attention, Concatenate, TimeDistributed, Dense, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore

class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder_input, decoder_input, input_vocab_size, target_vocab_size, latent_dim=50):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_input, input_vocab_size, latent_dim) # Encoder layer
        self.decoder = Decoder(decoder_input, target_vocab_size, latent_dim) # Decoder layer
        self.attn_layer = Attention() # Attention layer
        self.decoder_dense = TimeDistributed(Dense(target_vocab_size, activation='softmax')) # Dense layer
        

    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_outputs, state_h, state_c = self.encoder(encoder_input)
        decoder_outputs, _, _ = self.decoder([decoder_input, state_h, state_c])
        attn_out, _ = self.attn_layer([encoder_outputs, decoder_outputs])
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        decoder_outputs = self.decoder_dense(decoder_concat_input)
        return decoder_outputs
