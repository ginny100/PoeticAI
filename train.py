from data import Dataset
from model import RNN, Bidirectional
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model.beam_search import BeamSearch

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def train_model(dataset_path='dataset/tonghop.txt', model_dir='model', 
               epochs=50, batch_size=64, units=128, embedding_dim=256, dropout=0.3, learning_rate=0.001):
    """Train the poem generation model with the given parameters"""
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and process dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = Dataset(data_path=dataset_path)
    data_list = dataset.load_data()
    cleaned_data = dataset.clean_data()
    input_sentences, target_sentences = dataset.split_data()
    input_tokenizer, input_word2index, input_index2word = dataset.build_tokenizer(input_sentences)
    target_tokenizer, target_word2index, target_index2word= dataset.build_tokenizer(target_sentences)
    input_sequences, max_len_input_seq = dataset.tokenize(input_tokenizer, input_sentences)
    target_sequences, max_len_target_seq = dataset.tokenize(target_tokenizer, target_sentences)
    encoder_input_data, decoder_input_data, decoder_target_data = dataset.create_outputs(input_word2index, target_word2index, max_len_input_seq, max_len_target_seq)

    print("Step 2: Training model...")

    # Model hyperparameters
    LEARNING_RATE = learning_rate

    # Create checkpoint directory
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Define a simplified Seq2Seq model using Keras functional API
    encoder_inputs = tf.keras.layers.Input(shape=(None,), name="encoder_inputs")
    encoder_embedding = tf.keras.layers.Embedding(len(input_word2index) + 1, embedding_dim, name="encoder_embedding")(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(units, return_state=True, name="encoder_lstm")
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    # Decoder will use encoder's states as initial states
    decoder_inputs = tf.keras.layers.Input(shape=(None,), name="decoder_inputs")
    decoder_embedding = tf.keras.layers.Embedding(len(target_word2index) + 1, embedding_dim, name="decoder_embedding")(decoder_inputs)
    # Add dropout layer if specified
    if dropout > 0:
        decoder_embedding = tf.keras.layers.Dropout(dropout)(decoder_embedding)
    decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, name="decoder_lstm")(decoder_embedding, initial_state=[state_h, state_c])
    # Add dropout after LSTM if specified
    if dropout > 0:
        decoder_lstm = tf.keras.layers.Dropout(dropout)(decoder_lstm)
    decoder_outputs = tf.keras.layers.Dense(len(target_word2index) + 1, activation='softmax', name="decoder_dense")(decoder_lstm)

    # Create a combined model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    checkpoint_ext = ".keras"

    # Training using keras fit method
    history = model.fit(
        [encoder_input_data, decoder_input_data], 
        target_sequences,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"model_best{checkpoint_ext}"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max',
                verbose=1
            )
        ]
    )

    final_model_path = os.path.join(model_dir, 'final_model.keras')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot training metrics
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])

    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])

    plt.tight_layout()
    metrics_path = os.path.join(model_dir, 'training_metrics.png')
    plt.savefig(metrics_path)
    print(f"Training metrics saved to {metrics_path}")

    print("Training completed!")

    # Create inference models after training is complete
    # Define new input tensors for the encoder and decoder
    encoder_model_inputs = tf.keras.layers.Input(shape=(None,), name="inference_encoder_inputs")
    encoder_model_embedding = tf.keras.layers.Embedding(len(input_word2index) + 1, embedding_dim, name="inference_encoder_embedding")(encoder_model_inputs)
    encoder_model_lstm = tf.keras.layers.LSTM(units, return_state=True, name="inference_encoder_lstm")
    _, inference_state_h, inference_state_c = encoder_model_lstm(encoder_model_embedding)

    # Create the encoder model for inference
    encoder_model = tf.keras.Model(encoder_model_inputs, [inference_state_h, inference_state_c])

    # Set up the decoder model for inference
    decoder_state_input_h = tf.keras.layers.Input(shape=(units,), name="decoder_state_h_input")
    decoder_state_input_c = tf.keras.layers.Input(shape=(units,), name="decoder_state_c_input")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_inputs_inference = tf.keras.layers.Input(shape=(None,), name="inference_decoder_inputs")
    decoder_embedding_inference = tf.keras.layers.Embedding(len(target_word2index) + 1, embedding_dim, name="inference_decoder_embedding")(decoder_inputs_inference)
    if dropout > 0:
        decoder_embedding_inference = tf.keras.layers.Dropout(dropout)(decoder_embedding_inference)

    decoder_lstm_inference = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, name="inference_decoder_lstm")
    decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm_inference(
        decoder_embedding_inference, initial_state=decoder_states_inputs)
    
    if dropout > 0:
        decoder_outputs_inference = tf.keras.layers.Dropout(dropout)(decoder_outputs_inference)

    decoder_dense_inference = tf.keras.layers.Dense(len(target_word2index) + 1, activation='softmax', name="inference_decoder_dense")
    decoder_outputs_inference = decoder_dense_inference(decoder_outputs_inference)

    decoder_model = tf.keras.Model(
        [decoder_inputs_inference] + decoder_states_inputs,
        [decoder_outputs_inference, state_h_inference, state_c_inference]
    )

    # Save best model's encoder and decoder
    print("Copying the best model for inference...")
    best_model_path = os.path.join(checkpoint_dir, f"model_best{checkpoint_ext}")

    encoder_model_path = os.path.join(model_dir, 'encoder_model.keras')
    decoder_model_path = os.path.join(model_dir, 'decoder_model.keras')
    
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        best_model = tf.keras.models.load_model(best_model_path)
        
        # Use the best model's weights for inference models
        encoder_model.save(encoder_model_path)
        decoder_model.save(decoder_model_path)
    else:
        print(f"Warning: Best model not found at {best_model_path}, using final model instead")
        encoder_model.save(encoder_model_path)
        decoder_model.save(decoder_model_path)
    
    # Save tokenizer information as well
    with open(os.path.join(model_dir, 'tokenizer_info.txt'), 'w') as f:
        f.write(f"Dataset path: {dataset_path}\n")
        f.write(f"Input vocabulary size: {len(input_word2index) + 1}\n")
        f.write(f"Target vocabulary size: {len(target_word2index) + 1}\n")
        f.write(f"Max input sequence length: {max_len_input_seq}\n")
        f.write(f"Max target sequence length: {max_len_target_seq}\n")
    
    print(f"Model training complete. Encoder and decoder models saved to {model_dir}")
    return model, encoder_model, decoder_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an LSTM-based poem generation model')
    parser.add_argument('--dataset', type=str, default='dataset/tonghop.txt',
                        help='Path to the dataset file (default: dataset/tonghop.txt)')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory to save the model (default: model)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--units', type=int, default=128,
                        help='Number of LSTM units (default: 128)')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Dimension of the embedding layer (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer (default: 0.001)')
    
    args = parser.parse_args()
    
    # Train the model with the specified parameters
    train_model(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()

