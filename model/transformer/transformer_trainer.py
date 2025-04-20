import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from model.transformer_class import create_transformer, create_optimizer


class TransformerTrainer:
    """
    A class to handle training and evaluation of the Transformer model
    """
    def __init__(self, transformer, optimizer, loss_fn):
        self.transformer = transformer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # Set up metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_accuracy')
        
        # Checkpointing
        self.checkpoint_path = './checkpoints/transformer'
        self.ckpt = tf.train.Checkpoint(transformer=transformer,
                                       optimizer=optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, 
                                                      self.checkpoint_path, 
                                                      max_to_keep=5)
        
        # Load checkpoint if available
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Checkpoint restored from {}'.format(
                self.ckpt_manager.latest_checkpoint))
    
    def loss_function(self, real, pred):
        """Calculate loss with padding mask"""
        # Create mask for padding
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        
        # Apply mask to loss
        loss_ = self.loss_fn(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)
    
    @tf.function
    def train_step(self, inp, tar):
        """Execute one training step"""
        # Teacher forcing: use tar_inp as decoder input, tar_real for loss
        tar_inp = tar[:, :-1]  # Remove last token
        tar_real = tar[:, 1:]  # Remove first token (start)
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions, _ = self.transformer([inp, tar_inp], 
                                             training=True)
            
            # Calculate loss
            loss = self.loss_function(tar_real, predictions)
        
        # Calculate gradients and apply
        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, 
                                         self.transformer.trainable_variables))
        
        # Update metrics
        self.train_loss(loss)
        self.train_accuracy(tar_real, predictions)
        
        return loss
    
    @tf.function
    def val_step(self, inp, tar):
        """Execute one validation step"""
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        # Forward pass (no training)
        predictions, _ = self.transformer([inp, tar_inp], 
                                         training=False)
        
        # Calculate loss
        loss = self.loss_function(tar_real, predictions)
        
        # Update metrics
        self.val_loss(loss)
        self.val_accuracy(tar_real, predictions)
        
        return loss
    
    def train(self, train_dataset, val_dataset, epochs):
        """Train the model for a number of epochs"""
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            start = time.time()
            
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            
            # Training loop
            for (batch, (inp, tar)) in enumerate(train_dataset):
                loss = self.train_step(inp, tar)
                
                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {loss:.4f}')
            
            # Validation loop
            for (inp, tar) in val_dataset:
                self.val_step(inp, tar)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint at {ckpt_save_path}')
            
            # Print epoch results
            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} '
                  f'Accuracy {self.train_accuracy.result():.4f}')
            print(f'Validation Loss {self.val_loss.result():.4f} '
                  f'Validation Accuracy {self.val_accuracy.result():.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
            
            # Record history
            history['train_loss'].append(self.train_loss.result().numpy())
            history['train_accuracy'].append(self.train_accuracy.result().numpy())
            history['val_loss'].append(self.val_loss.result().numpy())
            history['val_accuracy'].append(self.val_accuracy.result().numpy())
        
        return history
    
    def plot_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs Epochs')
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='train_accuracy')
        plt.plot(history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy vs Epochs')
        
        plt.tight_layout()
        plt.savefig('transformer_training_history.png')
        plt.show()


def preprocess_dataset(dataset, buffer_size, batch_size, max_length):
    """Preprocess a dataset for transformer training"""
    # Filter by length
    dataset = dataset.filter(lambda x, y: tf.size(x) <= max_length and 
                           tf.size(y) <= max_length)
    
    # Shuffle, batch, and prefetch
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
