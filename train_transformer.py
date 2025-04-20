import tensorflow as tf
import numpy as np
import os
import argparse
import pickle
from data import Dataset
from model.transformer.transformer_class import create_transformer, create_optimizer

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def train_poem_transformer(dataset_path, epochs=50, batch_size=64, d_model=128, num_heads=8, 
                          num_layers=4, dff=512, dropout_rate=0.1, save_dir="saved_model/poem_transformer"):
    """Train a transformer model for poem generation"""
    print("\nLoading and preprocessing data...")
    # Prepare dataset
    dataset = Dataset(data_path=dataset_path)
    data_list = dataset.load_data()
    cleaned_data = dataset.clean_data()
    input_sentences, target_sentences = dataset.split_data()
    input_tokenizer, input_word2index, input_index2word = dataset.build_tokenizer(input_sentences)
    target_tokenizer, target_word2index, target_index2word = dataset.build_tokenizer(target_sentences)
    input_sequences, max_len_input_seq = dataset.tokenize(input_tokenizer, input_sentences)
    target_sequences, max_len_target_seq = dataset.tokenize(target_tokenizer, target_sentences)
    
    # Create directories for saving
    os.makedirs(save_dir, exist_ok=True)
    
    # Save tokenizers and vocabulary info using pickle (most reliable method)
    tokenizer_data = {
        'input_tokenizer': input_tokenizer,
        'input_word2index': input_word2index,
        'input_index2word': input_index2word,
        'target_tokenizer': target_tokenizer,
        'target_word2index': target_word2index,
        'target_index2word': target_index2word,
        'max_len_input_seq': max_len_input_seq,
        'max_len_target_seq': max_len_target_seq,
        'dataset_path': dataset_path
    }
    with open(f'{save_dir}/tokenizers.pickle', 'wb') as handle:
        pickle.dump(tokenizer_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Also save as text for human readability
    with open(f'{save_dir}/vocab_info.txt', 'w') as f:
        input_vocab_size = len(input_word2index) + 1
        target_vocab_size = len(target_word2index) + 1
        f.write(f"Input vocabulary size: {input_vocab_size}\n")
        f.write(f"Target vocabulary size: {target_vocab_size}\n")
        f.write(f"Max input sequence length: {max_len_input_seq}\n")
        f.write(f"Max target sequence length: {max_len_target_seq}\n")
        f.write(f"Dataset path: {dataset_path}\n")
    
    print(f"Input vocabulary size: {input_vocab_size}")
    print(f"Target vocabulary size: {target_vocab_size}")
    
    # Create TensorFlow datasets
    buffer_size = len(input_sequences)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (input_sequences, target_sequences))
    train_dataset = train_dataset.shuffle(buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    
    # Create transformer model
    print("\nCreating transformer model...")
    transformer = create_transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        pe_input=max_len_input_seq,
        pe_target=max_len_target_seq,
        dropout_rate=dropout_rate
    )
    
    # Create custom learning rate schedule
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            
            self.warmup_steps = warmup_steps
            
        def __call__(self, step):
            # Cast step to float32
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    # Create learning rate schedule and optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # Define loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    # Define metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    # Create checkpoint manager - use dataset-specific directory
    # Extract dataset name from path for the checkpoint directory
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    checkpoint_path = f'./checkpoints/poem_transformer_{dataset_name}'
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Using checkpoint directory: {checkpoint_path}")
    
    # Only checkpoint the model weights, not the optimizer state
    ckpt = tf.train.Checkpoint(model=transformer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    
    # Load checkpoint if available
    if ckpt_manager.latest_checkpoint:
        try:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print(f'\nCheckpoint restored from {ckpt_manager.latest_checkpoint}')
        except Exception as e:
            print(f"Warning: Error restoring checkpoint: {e}")
            print("Starting with fresh weights")
    
    # Define training step
    @tf.function
    def train_step(inp, tar):
        # Teacher forcing - feeding the target as the next input
        tar_inp = tar[:, :-1]  # Remove the last token (end)
        tar_real = tar[:, 1:]  # Remove the first token (start)
        
        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True)
            loss = loss_function(tar_real, predictions)
        
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(tar_real, predictions)
    
    # Training loop
    print("\nStarting training...")
    history = {
        'loss': [],
        'accuracy': []
    }
    
    for epoch in range(epochs):
        train_loss.reset_state()
        train_accuracy.reset_state()
        
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)
            
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint at {ckpt_save_path}')
        
        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        
        # Store metrics for plotting
        history['loss'].append(train_loss.result().numpy())
        history['accuracy'].append(train_accuracy.result().numpy())
    
    # Save the final model
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Save model in SavedModel format
        transformer.save(f'{save_dir}/model', save_format='tf')
        print(f"Model saved to {save_dir}/model")
    except Exception as e:
        print(f"Warning: Error saving model: {e}")
        # Fallback to weights-only saving
        transformer.save_weights(f'{save_dir}/transformer.weights.h5')
        print(f"Model weights saved to {save_dir}/transformer.weights.h5")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png')
        print(f"Training history plot saved to {save_dir}/training_history.png")
    except ImportError:
        print("Matplotlib not available, skipping training history plot")
    except Exception as e:
        print(f"Error creating training history plot: {e}")
    
    print(f"\nTraining completed! Model saved to {save_dir}")
    return transformer, input_tokenizer, target_tokenizer, input_word2index, target_word2index, input_index2word, target_index2word

def main():
    """Main function to handle command-line arguments for training"""
    parser = argparse.ArgumentParser(description='Train a transformer model for poem generation')
    parser.add_argument('--dataset', type=str, default='dataset/tonghop.txt',
                       help='Path to the dataset file')
    parser.add_argument('--model_dir', type=str, default='saved_model/poem_transformer',
                       help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Dimension of model')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--dff', type=int, default=512,
                       help='Dimension of feed forward network')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    
    args = parser.parse_args()
    
    # Train the model
    train_poem_transformer(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dff=args.dff,
        dropout_rate=args.dropout_rate,
        save_dir=args.model_dir
    )

if __name__ == "__main__":
    main() 