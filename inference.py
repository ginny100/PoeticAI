import tensorflow as tf
import numpy as np
import argparse
import os
from data import Dataset
from model.beam_search import BeamSearch
import matplotlib.pyplot as plt

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_models(checkpoint_dir):
    """
    Load encoder and decoder models from checkpoint directory
    """

    ext = ".keras"
    
    encoder_path = os.path.join(checkpoint_dir, f"encoder_model{ext}")
    decoder_path = os.path.join(checkpoint_dir, f"decoder_model{ext}")
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder model not found at {encoder_path}")
    
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder model not found at {decoder_path}")
    
    print(f"Loading encoder from: {encoder_path}")
    print(f"Loading decoder from: {decoder_path}")
    
    # Load models
    encoder_model = tf.keras.models.load_model(encoder_path)
    decoder_model = tf.keras.models.load_model(decoder_path)
    
    return encoder_model, decoder_model

def generate_fixed_length_poem_greedy(encoder_model, decoder_model, input_text, 
                              input_tokenizer, target_word2index, target_index2word,
                              max_len_input_seq, output_length=8):
    """
    Generate a poem with exactly output_length words using greedy search
    """
    # Check input text
    input_words = input_text.split()
    if len(input_words) != 6:
        print(f"Warning: Input should be 6 words. You provided {len(input_words)} words.")
    
    # Process input
    input_seq = input_tokenizer.texts_to_sequences([input_text])[0]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len_input_seq, padding='pre')
    
    # Encode the input
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Start with start token
    target_seq = np.zeros((1, 1))
    start_token_idx = target_word2index.get('start', 1)  # Default to 1 if 'start' not found
    target_seq[0, 0] = start_token_idx
    
    # Generate exactly output_length words
    generated_words = []
    
    while len(generated_words) < output_length:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Get the token with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = target_index2word.get(sampled_token_index, '')
        
        # Stop if we hit end token or unknown token
        if not sampled_token or sampled_token == 'end':
            break
        
        # Add the word to our results
        generated_words.append(sampled_token)
        
        # Update for next prediction
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    
    # Create final poem
    if len(generated_words) < output_length:
        print(f"Warning: Only generated {len(generated_words)} words instead of {output_length}")
    
    return ' '.join(generated_words)

def generate_variable_length_poem_greedy(encoder_model, decoder_model, input_text, 
                                 input_tokenizer, target_word2index, target_index2word,
                                 max_len_input_seq, max_length=50):
    """
    Generate a poem with variable length using greedy search
    """
    # Process input
    input_seq = input_tokenizer.texts_to_sequences([input_text])[0]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len_input_seq, padding='pre')
    
    # Encode the input
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Start with start token
    target_seq = np.zeros((1, 1))
    start_token_idx = target_word2index.get('start', 1)  # Default to 1 if 'start' not found
    target_seq[0, 0] = start_token_idx
    
    # Sampling loop
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = target_index2word.get(sampled_token_index, '')
        
        if sampled_token and sampled_token != 'end':
            decoded_sentence += ' ' + sampled_token
        
        # Exit condition: either hit max length or find stop token
        if sampled_token == 'end' or len(decoded_sentence.split()) > max_length:
            stop_condition = True
        
        # Update the target sequence (length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return decoded_sentence.strip()

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
    
    def greedy_decode(self, input_sentence, temperature=1.0, tokenizer=None, detokenizer=None):
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
        
        # Initialize with start token
        output = tf.convert_to_tensor([[self.start_token]])
        
        # Create encoder padding mask
        encoder_padding_mask = self.create_padding_mask(encoder_input)
        
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
            
            # Get the predicted token (greedy)
            predicted_id = tf.argmax(predictions, axis=-1)
            
            # Concatenate the predicted token to output
            output = tf.concat([output, predicted_id], axis=-1)
            
            # Exit if end token or max length
            if predicted_id == self.end_token:
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
        
        # Repeat encoder input for each beam
        encoder_input = tf.tile(encoder_input, [beam_width, 1])
        encoder_padding_mask = tf.tile(encoder_padding_mask, [beam_width, 1, 1, 1])
        
        # Initialize beam state
        beams = [(tf.convert_to_tensor([[self.start_token]]), 0.0)]
        completed_beams = []
        
        # Loop until max length or all beams complete
        for i in range(self.max_length):
            candidates = []
            
            # Expand each beam
            for j, (beam, score) in enumerate(beams):
                # If beam is completed, add to completed beams
                if beam[0, -1] == self.end_token:
                    completed_beams.append((beam, score))
                    continue
                
                # Create look ahead mask for decoder
                look_ahead_mask = self.create_look_ahead_mask(tf.shape(beam)[1])
                
                # Run the transformer
                predictions, _ = self.transformer(
                    [encoder_input[j:j+1], beam],
                    training=False
                )
                
                # Get the last token prediction
                predictions = predictions[0, -1, :]  # (vocab_size)
                
                # Get top k predictions
                top_k_values, top_k_indices = tf.math.top_k(predictions, k=beam_width)
                
                # Add to candidates
                for k in range(beam_width):
                    token_id = top_k_indices[k]
                    token_score = top_k_values[k]
                    
                    # Create new beam by appending token
                    new_beam = tf.concat([beam, [[token_id]]], axis=1)
                    
                    # Length normalization
                    new_score = score + tf.math.log(token_score)
                    new_score = new_score / ((5 + tf.shape(new_beam)[1]) ** alpha / 
                                           (5 + 1) ** alpha)
                    
                    candidates.append((new_beam, new_score))
            
            # Stop if no more candidates
            if not candidates:
                break
            
            # Sort candidates by score and keep top beam_width
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]
            
            # Early stopping if all beams end with end token
            if all(beam[0, -1] == self.end_token for beam, _ in beams):
                break
        
        # Add remaining beams to completed beams
        completed_beams.extend(beams)
        
        # Sort completed beams and choose the best one
        completed_beams.sort(key=lambda x: x[1], reverse=True)
        best_beam, _ = completed_beams[0]
        
        # Remove start token and end token if present
        output = best_beam[0, 1:]
        if output[-1] == self.end_token:
            output = output[:-1]
        
        if detokenizer:
            # Convert token IDs to text
            output_text = detokenizer(output.numpy())
            return output_text
        else:
            return output.numpy()
    
    def plot_attention_weights(self, attention_weights, layer_names=None, head_index=0):
        """Visualize attention weights from the model"""
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

def main():
    parser = argparse.ArgumentParser(description='Generate poems using trained models')
    parser.add_argument('--input', type=str, required=True, help='Input text for poem generation')
    parser.add_argument('--mode', type=str, default='fixed', choices=['fixed', 'variable'], 
                        help='Generation mode: fixed (8 words) or variable length')
    parser.add_argument('--output_length', type=int, default=8, 
                        help='Number of words to generate in fixed mode')
    parser.add_argument('--max_length', type=int, default=50,
                        help='Maximum number of words in variable mode')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory containing encoder and decoder models')
    parser.add_argument('--dataset_path', type=str, 
                        default='dataset/truyenkieu.txt',
                        help='Path to dataset for tokenizer creation')
    parser.add_argument('--search', type=str, default='beam', choices=['beam', 'greedy'],
                        help='Search strategy: beam or greedy')
    parser.add_argument('--beam_width', type=int, default=5,
                        help='Beam width for beam search (ignored for greedy search)')
    
    args = parser.parse_args()
    
    # Prepare dataset and tokenizers
    print("Loading and preprocessing data...")
    dataset = Dataset(data_path=args.dataset_path)
    
    # Try to use the existing dataset file
    if os.path.exists(args.dataset_path):
        dataset.dataPath = args.dataset_path
    else:
        # Or download it if not available
        dataset.download(url='https://raw.githubusercontent.com/tiensu/Natural_Language_Processing/master/Text-Generation/dataset/truyenkieu.txt')
    
    data_list = dataset.load_data()
    cleaned_data = dataset.clean_data()
    input_sentences, target_sentences = dataset.split_data()
    
    # Create tokenizers
    input_tokenizer, input_word2index, input_index2word = dataset.build_tokenizer(input_sentences)
    target_tokenizer, target_word2index, target_index2word = dataset.build_tokenizer(target_sentences)
    
    # Get sequence lengths
    input_sequences, max_len_input_seq = dataset.tokenize(input_tokenizer, input_sentences)
    target_sequences, max_len_target_seq = dataset.tokenize(target_tokenizer, target_sentences)
    
    # Load models
    print("Loading models...")
    encoder_model, decoder_model = load_models(args.model_dir)
    
    # Preprocess input
    input_seq = input_tokenizer.texts_to_sequences([args.input])[0]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len_input_seq, padding='pre')
    
    # Generate poem based on search strategy
    if args.search == 'beam':
        print(f"\nUsing beam search with beam width {args.beam_width}")
        beam_search = BeamSearch(encoder_model, decoder_model, target_word2index, target_index2word)
        
        if args.mode == 'fixed':
            poem = beam_search.generate_fixed_length_poem(input_seq, args.output_length, args.beam_width)
            print(f"\nInput: {args.input}")
            print(f"Generated poem ({args.output_length} words): {poem}")
        else:
            poem = beam_search.generate_poem(input_seq, args.max_length, args.beam_width)
            print(f"\nInput: {args.input}")
            print(f"Generated poem (variable length): {poem}")
    else:
        print("Using greedy search")
        if args.mode == 'fixed':
            poem = generate_fixed_length_poem_greedy(
                encoder_model, decoder_model, args.input,
                input_tokenizer, target_word2index, target_index2word,
                max_len_input_seq, args.output_length
            )
            print(f"\nInput: {args.input}")
            print(f"Generated poem ({args.output_length} words): {poem}")
        else:
            poem = generate_variable_length_poem_greedy(
                encoder_model, decoder_model, args.input,
                input_tokenizer, target_word2index, target_index2word,
                max_len_input_seq, args.max_length
            )
            print(f"\nInput: {args.input}")
            print(f"Generated poem (variable length): {poem}")

if __name__ == "__main__":
    main() 