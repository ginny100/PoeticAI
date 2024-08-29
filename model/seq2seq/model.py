import tensorflow as tf

from model.layer import Decoder, Encoder
from tensorflow.keras.layers import Attention, Concatenate, Dense, Input, TimeDistributed # type: ignore
from tensorflow.keras.models import Model # type: ignore

class Seq2Seq(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, latent_dim=50):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_vocab_size, latent_dim) # Encoder layer
        # print("Seq2Seq - input_vocab_size", input_vocab_size) # 1866
        self.decoder = Decoder(target_vocab_size, latent_dim) # Decoder layer
        # print("Seq2Seq - target_vocab_size", target_vocab_size) # 1987
        self.attn_layer = Attention() # Attention layer
        self.decoder_dense = TimeDistributed(Dense(target_vocab_size, activation='softmax')) # Dense layer wrapped with TimeDistributed
    
    @tf.function
    def call(self, inputs):
        encoder_input, decoder_input = inputs
        # print("Seq2Seq - encoder_input", encoder_input.shape) # (None, 6) -> (None, train - max_len_input_seq)
        # print("Seq2Seq - decoder_input", decoder_input.shape) # (None, 10) -> (None, train - max_len_target_seq)
        encoder_outputs, state_h, state_c = self.encoder(encoder_input)
        # print("Seq2Seq - hidden state - state_h", state_h) # (None, 50) -> (None, latent_dim)
        # print("Seq2Seq - cell state - state_c", state_c) # (None, 50) -> (None, latent_dim)
        decoder_outputs, _, _ = self.decoder(decoder_input, state_h, state_c)
        attn_out, attn_scores = self.attn_layer([decoder_outputs, encoder_outputs], return_attention_scores=True)
        # print("Seq2Seq - encoder_outputs", encoder_outputs.shape) # (None, 6, 50) -> (None, train - max_len_input_seq, latent_dim)
        # print("Seq2Seq - decoder_outputs", decoder_outputs.shape) # (None, 10, 50) -> (None, train - max_len_target_seq, latent_dim)
        # print("Seq2Seq - attn_out shape", attn_out.shape) # (None, 10, 50) -> (None, train - max_len_target_seq, latent_dim)
        # print("Seq2Seq - attn_scores shape", attn_scores.shape) # (None, 10, 6) -> (None, train - max_len_target_seq, train - max_len_input_seq)
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        # print("Seq2Seq - decoder_concat_input", decoder_concat_input.shape) # (None, 10, 100) -> (None, train - max_len_target_seq, latent_dim x 2)
        return self.decoder_dense(decoder_concat_input)

"""
To fit decoder_concat_input with shape (None, 10, 100) 
through the dense layer self.decoder_dense = Dense(target_vocab_size, activation='softmax') 
with target_vocab_size=1987, 
you need to ensure that the dense layer is applied to each time step individually by using the TimeDistributed wrapper, 
which applies a layer to every temporal slice of an input.
"""