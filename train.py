from data import Dataset
from model import RNN, Bidirectional
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from beam_search import BeamSearch

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

dataset = Dataset(data_path='dataset/tonghop.txt')
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
EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 256
UNITS = 128

# Create checkpoint directory
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Define a simplified Seq2Seq model using Keras functional API
encoder_inputs = tf.keras.layers.Input(shape=(None,), name="encoder_inputs")
encoder_embedding = tf.keras.layers.Embedding(len(input_word2index) + 1, EMBEDDING_SIZE, name="encoder_embedding")(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(UNITS, return_state=True, name="encoder_lstm")
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Decoder will use encoder's states as initial states
decoder_inputs = tf.keras.layers.Input(shape=(None,), name="decoder_inputs")
decoder_embedding = tf.keras.layers.Embedding(len(target_word2index) + 1, EMBEDDING_SIZE, name="decoder_embedding")(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(UNITS, return_sequences=True, name="decoder_lstm")(decoder_embedding, initial_state=[state_h, state_c])
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

# # Debug: Check shapes before training
# print(f"Encoder input shape: {encoder_input_data.shape}")
# print(f"Decoder input shape: {decoder_input_data.shape}")
# print(f"Target sequences shape: {target_sequences.shape}")

checkpoint_ext = ".keras"

# Training using keras fit method
history = model.fit(
    [encoder_input_data, decoder_input_data], 
    target_sequences,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
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

model.save('model/final_model.keras')

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
plt.savefig('training_metrics.png')
plt.show()

print("Training completed!")

# Create inference models after training is complete
# Define new input tensors for the encoder and decoder
encoder_model_inputs = tf.keras.layers.Input(shape=(None,), name="inference_encoder_inputs")
encoder_model_embedding = tf.keras.layers.Embedding(len(input_word2index) + 1, EMBEDDING_SIZE, name="inference_encoder_embedding")(encoder_model_inputs)
encoder_model_lstm = tf.keras.layers.LSTM(UNITS, return_state=True, name="inference_encoder_lstm")
_, inference_state_h, inference_state_c = encoder_model_lstm(encoder_model_embedding)

# Create the encoder model for inference
encoder_model = tf.keras.Model(encoder_model_inputs, [inference_state_h, inference_state_c])

# Set up the decoder model for inference
decoder_state_input_h = tf.keras.layers.Input(shape=(UNITS,), name="decoder_state_h_input")
decoder_state_input_c = tf.keras.layers.Input(shape=(UNITS,), name="decoder_state_c_input")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_inference = tf.keras.layers.Input(shape=(None,), name="inference_decoder_inputs")
decoder_embedding_inference = tf.keras.layers.Embedding(len(target_word2index) + 1, EMBEDDING_SIZE, name="inference_decoder_embedding")(decoder_inputs_inference)

decoder_lstm_inference = tf.keras.layers.LSTM(UNITS, return_sequences=True, return_state=True, name="inference_decoder_lstm")
decoder_outputs_inference, state_h_inference, state_c_inference = decoder_lstm_inference(
    decoder_embedding_inference, initial_state=decoder_states_inputs)

decoder_dense_inference = tf.keras.layers.Dense(len(target_word2index) + 1, activation='softmax', name="inference_decoder_dense")
decoder_outputs_inference = decoder_dense_inference(decoder_outputs_inference)

decoder_model = tf.keras.Model(
    [decoder_inputs_inference] + decoder_states_inputs,
    [decoder_outputs_inference, state_h_inference, state_c_inference]
)

# Save best model's encoder and decoder
print("Copying the best model for inference...")
best_model_path = os.path.join(checkpoint_dir, f"model_best{checkpoint_ext}")

if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    best_model = tf.keras.models.load_model(best_model_path)
    
    # Use the best model's weights for inference models
    encoder_model.save('model/encoder_model.keras')
    decoder_model.save('model/decoder_model.keras')
else:
    print(f"Warning: Best model not found at {best_model_path}, using final model instead")
    encoder_model.save('model/encoder_model.keras')
    decoder_model.save('model/decoder_model.keras')

