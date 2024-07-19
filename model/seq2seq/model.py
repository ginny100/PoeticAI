import tensorflow as tf

from model.layers import Decoder
from model.layers import Encoder

class Seq2Seq(tf.Keras.Model):
    def __init__(self, input_text_processor, output_text_processor, embedding_dim, units):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_text_processor.vocabulary_size(), embedding_dim, units)
        self.decoder = Decoder(output_text_processor.vocabulary_size(), embedding_dim, units)
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor