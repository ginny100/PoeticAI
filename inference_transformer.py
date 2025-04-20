import tensorflow as tf
import numpy as np
import os
import argparse
import pickle
from data import Dataset
from model.transformer.transformer_class import create_transformer, TransformerInference

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_poem_transformer(model_dir="saved_model/poem_transformer", dataset_path=None, use_checkpoint=True):
    """Load a trained transformer model for poem generation"""
    # First try to load the tokenizers from pickle (most reliable)
    tokenizer_data = None
    try:
        with open(f'{model_dir}/tokenizers.pickle', 'rb') as handle:
            tokenizer_data = pickle.load(handle)
        print("Successfully loaded tokenizers from pickle file")
        
        # Extract all necessary data
        input_tokenizer = tokenizer_data['input_tokenizer']
        input_word2index = tokenizer_data['input_word2index']
        input_index2word = tokenizer_data['input_index2word']
        target_tokenizer = tokenizer_data['target_tokenizer']
        target_word2index = tokenizer_data['target_word2index']
        target_index2word = tokenizer_data['target_index2word']
        max_len_input_seq = tokenizer_data['max_len_input_seq']
        max_len_target_seq = tokenizer_data['max_len_target_seq']
        saved_dataset_path = tokenizer_data.get('dataset_path', None)
        
        # Get vocabulary sizes
        input_vocab_size = len(input_word2index) + 1
        target_vocab_size = len(target_word2index) + 1
        
        print(f"Using tokenizers from saved model: input_vocab={input_vocab_size}, target_vocab={target_vocab_size}")
        
        # Use saved dataset path if available and not overridden
        if saved_dataset_path and not dataset_path:
            dataset_path = saved_dataset_path
            print(f"Using dataset path from saved tokenizers: {dataset_path}")
    except Exception as e:
        print(f"Could not load tokenizers from pickle: {e}")
        tokenizer_data = None
    
    # If pickle loading failed, try to load from text file
    if tokenizer_data is None:
        try:
            with open(f'{model_dir}/vocab_info.txt', 'r') as f:
                lines = f.readlines()
                input_vocab_size = int(lines[0].split(': ')[1])
                target_vocab_size = int(lines[1].split(': ')[1])
                max_len_input_seq = int(lines[2].split(': ')[1])
                max_len_target_seq = int(lines[3].split(': ')[1])
                
                if len(lines) > 4 and "Dataset path:" in lines[4] and not dataset_path:
                    saved_dataset_path = lines[4].split(': ')[1].strip()
                    dataset_path = saved_dataset_path
                    print(f"Using dataset from model info: {dataset_path}")
                
                print(f"Using vocab sizes from text file: input={input_vocab_size}, target={target_vocab_size}")
        except Exception as e:
            print(f"Warning: Could not load vocab_info.txt: {e}")
            input_vocab_size = None
            target_vocab_size = None
            max_len_input_seq = None
            max_len_target_seq = None
    
    # If we still don't have tokenizers, we need to rebuild them from the dataset
    if tokenizer_data is None:
        # Use the dataset path from the loaded info or the provided one
        dataset_to_use = dataset_path if dataset_path else 'dataset/truyenkieu.txt'
        print(f"Loading dataset from {dataset_to_use} to rebuild tokenizers")
        
        # Load the dataset and build tokenizers
        dataset = Dataset(data_path=dataset_to_use)
        data_list = dataset.load_data()
        cleaned_data = dataset.clean_data()
        input_sentences, target_sentences = dataset.split_data()
        input_tokenizer, input_word2index, input_index2word = dataset.build_tokenizer(input_sentences)
        target_tokenizer, target_word2index, target_index2word = dataset.build_tokenizer(target_sentences)
        
        # Get sequence lengths from the dataset
        input_sequences, dataset_max_len_input = dataset.tokenize(input_tokenizer, input_sentences)
        target_sequences, dataset_max_len_target = dataset.tokenize(target_tokenizer, target_sentences)
        
        # If we have saved vocab sizes, use them (important for loading the model)
        # Otherwise use the ones from the current dataset
        if input_vocab_size is None:
            input_vocab_size = len(input_word2index) + 1
        if target_vocab_size is None:
            target_vocab_size = len(target_word2index) + 1
        
        # For max lengths, prefer saved values but default to dataset values if needed
        if max_len_input_seq is None:
            max_len_input_seq = dataset_max_len_input 
        if max_len_target_seq is None:
            max_len_target_seq = dataset_max_len_target
    
    # Create transformer with the architecture matching the saved model
    print(f"Creating transformer with vocab sizes: input={input_vocab_size}, target={target_vocab_size}")
    transformer = create_transformer(
        num_layers=4,  # Default values, adjust if needed
        d_model=128,
        num_heads=8,
        dff=512,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=max_len_input_seq,
        pe_target=max_len_target_seq,
        dropout_rate=0.1
    )
    
    # Try loading from checkpoint first if requested
    loading_succeeded = False
    if use_checkpoint:
        # Use dataset-specific checkpoint directory if dataset path is provided
        if dataset_path:
            dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
            checkpoint_path = f'./checkpoints/poem_transformer_{dataset_name}'
        else:
            checkpoint_path = './checkpoints/poem_transformer'
            
        print(f"Looking for checkpoints in: {checkpoint_path}")
        ckpt = tf.train.Checkpoint(model=transformer)
        
        if os.path.exists(checkpoint_path):
            # Get the latest checkpoint
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            if latest_checkpoint:
                try:
                    print(f"Attempting to restore from checkpoint: {latest_checkpoint}")
                    ckpt.restore(latest_checkpoint).expect_partial()
                    print(f"Model loaded from checkpoint: {latest_checkpoint}")
                    loading_succeeded = True
                except Exception as e:
                    print(f"Error loading from checkpoint: {e}")
                    print("This could be due to vocabulary size mismatch. Training might be needed.")
    
    # If checkpoint loading failed or wasn't requested, try loading weights
    if not loading_succeeded:
        try:
            # Try loading weights directly
            transformer.load_weights(f'{model_dir}/transformer.weights.h5')
            print(f"Model weights loaded from {model_dir}/transformer.weights.h5")
            loading_succeeded = True
        except Exception as e:
            print(f"Error loading weights: {e}")
            try:
                # Fall back to loading SavedModel if weights fail
                transformer = tf.keras.models.load_model(f'{model_dir}/model')
                print(f"Full model loaded from {model_dir}/model")
                loading_succeeded = True
            except Exception as e:
                print(f"Error loading model from {model_dir}: {e}")
                print("Using freshly initialized model")
    
    return transformer, input_tokenizer, target_tokenizer, input_word2index, target_word2index, input_index2word, target_index2word

def generate_poem(transformer, input_text, input_tokenizer, target_tokenizer, 
                 target_word2index, target_index2word, max_len_input_seq, 
                 max_length=50, beam_width=5, temperature=0.7, max_words=8):
    """Generate a poem using the transformer model with beam search"""
    # Create inference object
    inference = TransformerInference(
        transformer=transformer,
        max_length=max_length,
        start_token=target_word2index.get('start', 1),
        end_token=target_word2index.get('end', 2)
    )
    
    # Process input
    input_seq = input_tokenizer.texts_to_sequences([input_text])[0]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len_input_seq, padding='post')
    
    # Define detokenizer function to limit words
    def detokenize(token_sequence):
        if len(token_sequence) == 0:
            return "Unable to generate text with current model"
        words = [target_index2word.get(idx, '') for idx in token_sequence if idx > 0]
        # Filter out special tokens
        words = [word for word in words if word not in ['start', 'end', '']]
        # Limit number of words
        words = words[:max_words]
        if not words:
            return "Unable to generate meaningful text"
        return ' '.join(words)
    
    # Try beam search first, fall back to greedy if it fails
    try:
        if beam_width > 1:
            output = inference.beam_search_decode(
                input_seq[0], 
                beam_width=beam_width,
                detokenizer=detokenize
            )
            return output
        else:
            # Generate with greedy search
            output_text, _ = inference.greedy_decode(
                input_seq[0],
                temperature=temperature,
                detokenizer=detokenize
            )
            return output_text
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Falling back to simple generation")
        
        # Simple fallback generation
        try:
            # Generate output using a single forward pass
            start_token_tensor = tf.convert_to_tensor([[target_word2index.get('start', 1)]], dtype=tf.int32)
            predictions, _ = transformer([input_seq, start_token_tensor], training=False)
            predicted_ids = tf.argmax(predictions[:, 0, :], axis=-1, output_type=tf.int32).numpy()
            
            # Convert to text
            if isinstance(predicted_ids, (list, np.ndarray)) and len(predicted_ids) > 0:
                predicted_id = predicted_ids[0]
                words = [target_index2word.get(predicted_id, 'unknown')]
                return ' '.join(words)
            else:
                return "Generation failed, try again with different parameters"
        except Exception as e:
            print(f"Fallback generation also failed: {e}")
            return "Unable to generate text with current model"

def direct_generation(transformer, input_text, input_tokenizer, target_tokenizer, 
                     target_word2index, target_index2word, max_len_input_seq, 
                     temperature=0.7, max_words=8):
    """Generate a poem directly by sampling from the model one token at a time"""
    # Process input text
    input_seq = input_tokenizer.texts_to_sequences([input_text])[0]
    input_seq = tf.keras.preprocessing.sequence.pad_sequences([input_seq], maxlen=max_len_input_seq, padding='post')
    
    # Initialize with start token
    start_token = target_word2index.get('start', 1)
    end_token = target_word2index.get('end', 2)
    output_tokens = [start_token]
    
    # Maximum length safety
    max_length = 50
    
    # Generate tokens one by one
    for i in range(max_length):
        # Convert to tensor
        decoder_input = tf.convert_to_tensor([output_tokens], dtype=tf.int32)
        
        # Get predictions
        predictions, _ = transformer([input_seq, decoder_input], training=False)
        
        # Get the last token prediction
        predictions = predictions[:, -1, :] 
        
        if temperature == 0:
            # Greedy selection
            predicted_id = tf.argmax(predictions, axis=-1).numpy()[0]
        else:
            # Apply temperature
            predictions = predictions / temperature
            # Remove potential for NaN or Inf
            predictions = tf.where(tf.math.is_nan(predictions), tf.zeros_like(predictions), predictions)
            predictions = tf.where(tf.math.is_inf(predictions), tf.zeros_like(predictions), predictions)
            # Sample from the distribution
            predicted_id = tf.random.categorical(predictions, num_samples=1).numpy()[0][0]
        
        # Break if we hit the end token
        if predicted_id == end_token:
            break
            
        # Add to output
        output_tokens.append(predicted_id)
        
        # Stop if we've reached the maximum number of words (not counting special tokens)
        words_generated = len([t for t in output_tokens if t > 2])  # Skip start and end tokens
        if words_generated >= max_words:
            break
    
    # Convert tokens to words
    words = []
    for token in output_tokens:
        if token > 2:  # Skip start and end tokens
            word = target_index2word.get(token, '')
            if word:  # Only add non-empty words
                words.append(word)
    
    if not words:
        return "Unable to generate text with current model"
        
    return ' '.join(words)

def main():
    """Main function to handle command-line arguments for inference"""
    parser = argparse.ArgumentParser(description='Vietnamese poem generation with a transformer model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input text for poem generation')
    parser.add_argument('--dataset', type=str, default='dataset/tonghop.txt',
                       help='Path to the dataset file for tokenizer creation')
    parser.add_argument('--model_dir', type=str, default='saved_model/poem_transformer',
                       help='Directory with pre-trained model')
    parser.add_argument('--beam_width', type=int, default=1,
                       help='Beam width for beam search (set to 1 for greedy search)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for sampling (only used with greedy search)')
    parser.add_argument('--max_length', type=int, default=50,
                       help='Maximum length of generated poem')
    parser.add_argument('--max_words', type=int, default=8,
                       help='Maximum number of words in the generated poem')
    parser.add_argument('--use_checkpoint', type=bool, default=True,
                       help='Use the latest checkpoint instead of saved model')
    parser.add_argument('--direct', action='store_true',
                       help='Use direct generation method (similar to training)')
    
    args = parser.parse_args()
    
    # Load the model - always use checkpoint by default for best results
    transformer, input_tokenizer, target_tokenizer, input_word2index, target_word2index, input_index2word, target_index2word = load_poem_transformer(
        model_dir=args.model_dir,
        dataset_path=args.dataset,
        use_checkpoint=args.use_checkpoint
    )
    
    # Try different approaches and show the results
    print(f"\nInput: {args.input}")
    
    if args.direct:
        # Use direct generation - similar to how the model was trained
        print("\nUsing direct generation (similar to training):")
        direct_poem = direct_generation(
            transformer=transformer,
            input_text=args.input,
            input_tokenizer=input_tokenizer,
            target_tokenizer=target_tokenizer,
            target_word2index=target_word2index,
            target_index2word=target_index2word,
            max_len_input_seq=max(50, len(input_tokenizer.word_index) + 1),  # Safe default
            temperature=args.temperature,
            max_words=args.max_words
        )
        print(f"Generated poem: {direct_poem}")
    else:
        # Generate with greedy search first (more reliable)
        print("\nUsing greedy search:")
        greedy_poem = generate_poem(
            transformer=transformer,
            input_text=args.input,
            input_tokenizer=input_tokenizer,
            target_tokenizer=target_tokenizer,
            target_word2index=target_word2index,
            target_index2word=target_index2word,
            max_len_input_seq=max(50, len(input_tokenizer.word_index) + 1),  # Safe default
            max_length=args.max_length,
            beam_width=1,
            temperature=args.temperature,
            max_words=args.max_words
        )
        print(f"Generated poem: {greedy_poem}")
        
        # Try with beam search if requested
        if args.beam_width > 1:
            print(f"\nUsing beam search (width={args.beam_width}):")
            beam_poem = generate_poem(
                transformer=transformer,
                input_text=args.input,
                input_tokenizer=input_tokenizer,
                target_tokenizer=target_tokenizer,
                target_word2index=target_word2index,
                target_index2word=target_index2word,
                max_len_input_seq=max(50, len(input_tokenizer.word_index) + 1),  # Safe default
                max_length=args.max_length,
                beam_width=args.beam_width,
                temperature=1.0,
                max_words=args.max_words
            )
            print(f"Generated poem: {beam_poem}")

if __name__ == "__main__":
    main() 