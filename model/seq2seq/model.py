import tensorflow as tf

from model.layer import Decoder
from model.layer import Encoder
from tensorflow.keras.layers import Attention, Concatenate, Dense, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore

class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder_input, decoder_input, input_vocab_size, target_vocab_size, latent_dim=50):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_vocab_size, latent_dim) # Encoder layer
        self.decoder = Decoder(target_vocab_size, latent_dim) # Decoder layer
        self.attn_layer = Attention() # Attention layer
        self.decoder_dense = Dense(target_vocab_size, activation='softmax') # Dense layer
    
    @tf.function
    def call(self, inputs):
        encoder_input, decoder_input = inputs
        encoder_outputs, state_h, state_c = self.encoder(encoder_input) # state_h.shape=(None, 1987) should be (None, latent_dim) = (None, 50)
        print("hidden", state_h) # 
        print("state_c", state_c)
        decoder_outputs, _, _ = self.decoder(decoder_input, state_h, state_c)
        # attn_out, attn_scores = self.attn_layer([encoder_outputs, decoder_outputs], return_attention_scores=True) # Why always need `return_attention_scores=True` here?
        attn_out, attn_scores = self.attn_layer([decoder_outputs, encoder_outputs], return_attention_scores=True) # Why always need `return_attention_scores=True` here?
        print("Encoder Out:", encoder_outputs) # (None, 6, 50)
        print("Decoder Out:", decoder_outputs) # (None, 10, 50)
        print("Attention Output Shape:", attn_out.shape) # (None, 6, 50) -> (None, max_len_input_seq, latent_dim)
        print("Attention Scores Shape:", attn_scores.shape)
        decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])
        return self.decoder_dense(decoder_concat_input)
