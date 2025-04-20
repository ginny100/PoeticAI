import tensorflow as tf
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for Transformer architecture.
    
    This layer adds positional information to the input embeddings
    to help the model understand the order of sequence elements.
    """
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def get_angles(self, position, i, d_model):
        """Calculate the angles for positional encoding"""
        # Cast both arguments to the same type (float32) to avoid type mismatch in tf.pow
        power_term = tf.cast(2 * (i // 2), tf.float32) / tf.cast(d_model, tf.float32)
        denominator = tf.pow(tf.cast(10000, tf.float32), power_term)
        angles = tf.cast(position, tf.float32) / denominator
        return angles
    
    def positional_encoding(self, position, d_model):
        """Generate positional encoding matrix"""
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        # Apply sine to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])
        
        # Apply cosine to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        """Add positional encoding to inputs"""
        input_shape = tf.shape(inputs)
        seq_len = input_shape[1]
        
        # Use tf.cond to handle the condition in a way compatible with both graph and eager execution
        def true_fn():
            # When sequence length is greater than position
            extended_pos_encoding = self.positional_encoding(seq_len, self.d_model)
            return inputs + extended_pos_encoding[:, :seq_len, :]
        
        def false_fn():
            # When sequence length is less than or equal to position
            return inputs + self.pos_encoding[:, :seq_len, :]
        
        return tf.cond(
            tf.greater(seq_len, self.position),
            true_fn,
            false_fn
        )


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer for Transformer architecture.
    
    This layer splits the representation into multiple heads to allow
    the model to jointly attend to information from different positions.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        # Add the mask to the scaled tensor
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights


class PositionwiseFeedForward(tf.keras.layers.Layer):
    """
    Position-wise Feed-Forward Network for Transformer architecture.
    
    This is a fully connected feed-forward network applied to each position separately.
    """
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        """Apply position-wise feed-forward network"""
        x = self.dense1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer for Transformer architecture.
    
    Each encoder layer consists of a multi-head attention layer
    followed by a position-wise feed-forward network.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout_rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        """Apply encoder layer to input"""
        attn_output, _ = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Add & Norm
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Add & Norm
        
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder layer for Transformer architecture.
    
    Each decoder layer consists of two multi-head attention layers
    followed by a position-wise feed-forward network.
    """
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout_rate)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        """Apply decoder layer to input"""
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # Masked self-attention
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # Cross-attention
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2


class TransformerEncoder(tf.keras.layers.Layer):
    """
    Transformer Encoder.
    
    Stack of encoder layers with embedding and positional encoding.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 maximum_position_encoding, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) 
                          for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        """Apply encoder to input sequence"""
        seq_len = tf.shape(x)[1]
        
        # Convert input indices to embeddings
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x, training=training)
        
        # Pass through each encoder layer
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        
        return x  # (batch_size, input_seq_len, d_model)


class TransformerDecoder(tf.keras.layers.Layer):
    """
    Transformer Decoder.
    
    Stack of decoder layers with embedding and positional encoding.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
                 maximum_position_encoding, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) 
                          for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        """Apply decoder to input sequence"""
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        # Convert target indices to embeddings
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x, training=training)
        
        # Pass through each decoder layer
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training=training,
                                                  look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        # (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    """
    Transformer model.
    
    Complete implementation of the Transformer architecture with 
    encoder, decoder, and final output layer.
    """
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, pe_input, pe_target, dropout_rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, 
                                          input_vocab_size, pe_input, dropout_rate)
        
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, dff, 
                                          target_vocab_size, pe_target, dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def create_masks(self, inp, tar):
        """Create masks for encoder and decoder"""
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)
        
        # Decoder padding mask for the encoder output
        dec_padding_mask = self.create_padding_mask(inp)
        
        # Look ahead mask and padding mask for the decoder inputs
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def create_padding_mask(self, seq):
        """Create mask for padding tokens"""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def create_look_ahead_mask(self, size):
        """Create mask to prevent attending to future tokens"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def call(self, inputs, training=False):
        """Apply transformer to inputs"""
        inp, tar = inputs
        
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training=training, look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights


def create_transformer(num_layers=6, d_model=512, num_heads=8, dff=2048, 
                       input_vocab_size=8500, target_vocab_size=8000, 
                       pe_input=10000, pe_target=6000, dropout_rate=0.1):
    """Create a transformer model with the specified parameters"""
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=pe_input,
        pe_target=pe_target,
        dropout_rate=dropout_rate
    )
    
    return transformer


def create_optimizer(d_model):
    """Create a custom learning rate scheduler and optimizer"""
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            
            self.warmup_steps = warmup_steps
            
        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                          epsilon=1e-9)
    
    return optimizer


class TransformerInference:
    """
    A class to handle inference with the Transformer model
    """
    def __init__(self, transformer, max_length, start_token=None, end_token=None):
        self.transformer = transformer
        self.max_length = max_length
        self.start_token = start_token if start_token is not None else 1
        self.end_token = end_token if end_token is not None else 2
    
    def encode_input(self, sentence, tokenizer=None):
        """Encode a sentence for inference"""
        if tokenizer:
            # Use tokenizer if provided
            tokens = tokenizer.texts_to_sequences([sentence])[0]
            return tf.expand_dims(tokens, 0)
        else:
            # Assume sentence is already tokenized
            return tf.expand_dims(sentence, 0)
    
    def create_padding_mask(self, seq):
        """Create mask for padding tokens"""
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def create_look_ahead_mask(self, size):
        """Create mask to prevent attending to future tokens"""
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)
    
    def greedy_decode(self, input_sentence, temperature=0.7, tokenizer=None, detokenizer=None):
        """
        Generate sequence using greedy decoding
        
        Args:
            input_sentence: Input sentence or token sequence
            temperature: Sampling temperature (1.0 means no temperature)
            tokenizer: Optional tokenizer for text input
            detokenizer: Optional detokenizer for text output
            
        Returns:
            The generated sequence
        """
        # Encode input
        encoder_input = self.encode_input(input_sentence, tokenizer)
        
        # Initialize with start token - ensure int32 type
        output = tf.convert_to_tensor([[self.start_token]], dtype=tf.int32)
        
        # Track attention weights for visualization
        attention_weights_all = {}
        
        # Loop until max length or end token
        for i in range(self.max_length):
            # Create look ahead mask for decoder
            look_ahead_mask = self.create_look_ahead_mask(tf.shape(output)[1])
            decoder_padding_mask = self.create_padding_mask(encoder_input)
            
            # Run the transformer
            predictions, attention_weights = self.transformer(
                [encoder_input, output],
                training=False
            )
            
            # Store attention weights for visualization
            for key, value in attention_weights.items():
                attention_weights_all[f"{key}_step{i}"] = value
            
            # Get the last token prediction
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            
            # Apply temperature scaling
            if temperature != 1.0:
                predictions = predictions / temperature
            
            # Sample from the distribution instead of taking argmax for diversity
            if temperature < 1.0:
                # Apply softmax to get a probability distribution
                predictions_probs = tf.nn.softmax(predictions, axis=-1)
                
                # Sample from the distribution
                predicted_id = tf.random.categorical(
                    tf.math.log(tf.reshape(predictions_probs, [1, -1])), 
                    num_samples=1,
                    dtype=tf.int32  # Ensure int32 output
                )
                predicted_id = tf.reshape(predicted_id, [1, 1])
            else:
                # Just use greedy (argmax) if temperature is 1.0
                predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)
                predicted_id = tf.expand_dims(predicted_id, 0)
            
            # Concatenate the predicted token to output
            output = tf.concat([output, predicted_id], axis=-1)
            
            # Exit if end token or max length
            if tf.cast(predicted_id[0, 0], tf.int32) == tf.cast(self.end_token, tf.int32):
                break
        
        # Convert to the final output format
        output = output[0, 1:]  # Remove the start token and batch dimension
        
        if detokenizer:
            # Convert token IDs to text
            output_text = detokenizer(output.numpy())
            return output_text, attention_weights_all
        else:
            return output.numpy(), attention_weights_all
    
    def beam_search_decode(self, input_sentence, beam_width=4, alpha=0.6, 
                          tokenizer=None, detokenizer=None):
        """
        Generate sequence using beam search decoding
        
        Args:
            input_sentence: Input sentence or token sequence
            beam_width: Number of beams to consider
            alpha: Length penalty parameter
            tokenizer: Optional tokenizer for text input
            detokenizer: Optional detokenizer for text output
            
        Returns:
            The best generated sequence
        """
        # Encode input
        encoder_input = self.encode_input(input_sentence, tokenizer)
        
        # Create encoder padding mask
        encoder_padding_mask = self.create_padding_mask(encoder_input)
        
        # Initialize beam state with int32 tensors
        beams = [(tf.convert_to_tensor([[self.start_token]], dtype=tf.int32), 0.0)]
        completed_beams = []
        
        # Convert end token to int32 for comparison
        end_token = tf.constant(self.end_token, dtype=tf.int32)
        
        # Loop until max length or all beams complete
        for i in range(self.max_length):
            candidates = []
            
            # Expand each beam
            for j, (beam, score) in enumerate(beams):
                # If beam is completed, add to completed beams
                if beam.shape[1] > 1 and tf.equal(beam[0, -1], end_token):
                    completed_beams.append((beam, score))
                    continue
                
                # Create look ahead mask for decoder
                look_ahead_mask = self.create_look_ahead_mask(tf.shape(beam)[1])
                
                # Run the transformer - use a single copy of encoder_input for each beam
                try:
                    predictions, _ = self.transformer(
                        [encoder_input, beam],
                        training=False
                    )
                except (tf.errors.InvalidArgumentError, ValueError) as e:
                    print(f"Error in beam {j} at step {i}. Beam shape: {beam.shape}, Input shape: {encoder_input.shape}")
                    print(e)
                    # Try to recover by skipping this beam
                    continue
                
                # Get the last token prediction
                predictions = predictions[0, -1, :]  # (vocab_size)
                
                # Get top k predictions
                top_k_values, top_k_indices = tf.math.top_k(predictions, k=beam_width)
                
                # Convert indices to int32
                top_k_indices = tf.cast(top_k_indices, tf.int32)
                
                # Add to candidates
                for k in range(beam_width):
                    token_id = top_k_indices[k]
                    token_score = top_k_values[k]
                    
                    # Create new beam by appending token - ensure both tensors are int32
                    token_id_tensor = tf.reshape(token_id, [1, 1])  # Shape [1, 1]
                    new_beam = tf.concat([beam, token_id_tensor], axis=1)
                    
                    # Length normalization
                    new_score = score + tf.math.log(token_score)
                    beam_length = tf.cast(tf.shape(new_beam)[1], tf.float32)
                    alpha_tensor = tf.cast(alpha, tf.float32)
                    new_score = new_score / ((5 + beam_length) ** alpha_tensor / 
                                           (5 + 1.0) ** alpha_tensor)
                    
                    candidates.append((new_beam, new_score))
            
            # Stop if no more candidates
            if not candidates:
                break
            
            # Sort candidates by score and keep top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            # Early stopping if all beams end with end token
            if all(tf.equal(beam[0, -1], end_token) for beam, _ in beams if beam.shape[1] > 0):
                break
        
        # Add remaining beams to completed beams
        completed_beams.extend(beams)
        
        # Sort completed beams and choose the best one
        if not completed_beams:
            # If no complete beams, return empty result
            if detokenizer:
                return ""
            else:
                return np.array([])
        
        completed_beams.sort(key=lambda x: x[1], reverse=True)
        best_beam, _ = completed_beams[0]
        
        # Handle empty case
        if best_beam.shape[1] <= 1:  # Only has start token or nothing
            if detokenizer:
                return ""
            else:
                return np.array([])
        
        # Remove start token and end token if present
        output = best_beam[0, 1:]  # Remove start token
        
        # Check if the last token is the end token, safely
        if output.shape[0] > 0 and tf.equal(output[-1], end_token):
            output = output[:-1]
        
        if detokenizer:
            # Convert token IDs to text
            output_text = detokenizer(output.numpy())
            return output_text
        else:
            return output.numpy()
    
    def plot_attention_weights(self, attention_weights, layer_names=None, head_index=0):
        """Visualize attention weights from the model"""
        import matplotlib.pyplot as plt
        
        # Collect all attention weight matrices
        attention_matrices = []
        
        # Get attention weights from specified layers, default to all layers
        if layer_names is None:
            layer_names = [key for key in attention_weights.keys() 
                          if 'decoder_layer' in key and 'block2' in key]
        
        # Extract attention weights for each layer
        for layer_name in layer_names:
            attention_matrices.append(attention_weights[layer_name][0, head_index, :, :])
        
        # Set up the figure
        fig, axes = plt.subplots(len(attention_matrices), 1, 
                                figsize=(12, 4 * len(attention_matrices)))
        
        # Handle single layer case
        if len(attention_matrices) == 1:
            axes = [axes]
        
        # Plot each attention matrix
        for i, attention_matrix in enumerate(attention_matrices):
            # Plot attention weights
            im = axes[i].imshow(attention_matrix, cmap='viridis')
            
            # Add colorbar
            fig.colorbar(im, ax=axes[i])
            
            # Add labels
            axes[i].set_title(f'Layer {i+1} Attention Weights (Head {head_index})')
            axes[i].set_xlabel('Input Sequence Position')
            axes[i].set_ylabel('Output Sequence Position')
        
        plt.tight_layout()
        plt.savefig('attention_weights.png')
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Sample parameters
    num_layers = 4
    d_model = 128
    num_heads = 8
    dff = 512
    input_vocab_size = 10000
    target_vocab_size = 10000
    dropout_rate = 0.1
    
    # Maximum sequence lengths
    pe_input = 100
    pe_target = 100
    
    # Create transformer
    transformer = create_transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=pe_input,
        pe_target=pe_target,
        dropout_rate=dropout_rate
    )
    
    # Create optimizer
    optimizer = create_optimizer(d_model)
    
    # Sample input (batch_size=2, seq_len=5)
    inp = tf.random.uniform((2, 5), dtype=tf.int64, minval=0, maxval=input_vocab_size)
    tar = tf.random.uniform((2, 5), dtype=tf.int64, minval=0, maxval=target_vocab_size)
    
    # Forward pass
    output, _ = transformer([inp, tar], training=False)
    
    print(f"Input shape: {inp.shape}")
    print(f"Target shape: {tar.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check how many parameters
    print(f"Total parameters: {transformer.count_params():,}")
    
    # Create a simple model summary
    transformer.summary() 