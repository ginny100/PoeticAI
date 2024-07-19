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
    
    @tf.function
    def train_step(self, data):
        input_sentence, output_sentence = data
        input_word_indices = self.input_text_processor(input_sentence)
        output_word_indices = self.output_text_processor(output_sentence)
        output_mask = tf.cast(output_word_indices!=0, tf.float32)
        
        with tf.GradientTape() as tape:
            whole_encoder_states, final_hidden_state = self.encoder(input_word_indices)
            vocab_output, decoder_last_state = self.decoder(
                input_word_indices=output_word_indices,
                encoder_keys=whole_encoder_states,
                mask=(input_word_indices != 0),
                state=final_hidden_state
            )
        
        loss = tf.reduce_sum(self.loss(y_true=output_word_indices[:, 1:], y_pred=vocab_output[:, :-1])*output_mask[:, 1:])
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        return self._step(data, training=True)
