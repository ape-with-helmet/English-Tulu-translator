import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use("TkAgg")  # Or "Qt5Agg" if TkAgg is not available

import matplotlib.pyplot as plt

import pickle
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

data = pd.read_csv('translation_train_dataset.csv')

# Display the first few rows
print(data.head())

# Parameters
max_length = 20  # Maximum length of sequences
num_samples = len(data)  # Total number of samples

# Initialize Tokenizer
english_tokenizer = Tokenizer()
tulu_tokenizer = Tokenizer()

# Fit on English and Tulu sentences
english_tokenizer.fit_on_texts(data['English'])
tulu_tokenizer.fit_on_texts(data['Tulu'])

# Save the tokenizers
with open('english_tokenizer.pkl', 'wb') as file:
    pickle.dump(english_tokenizer, file)

with open('tulu_tokenizer.pkl', 'wb') as file:
    pickle.dump(tulu_tokenizer, file)

# Convert sentences to sequences
english_sequences = english_tokenizer.texts_to_sequences(data['English'])
tulu_sequences = tulu_tokenizer.texts_to_sequences(data['Tulu'])

# Pad sequences
english_sequences = pad_sequences(english_sequences, maxlen=max_length, padding='post')
tulu_sequences = pad_sequences(tulu_sequences, maxlen=max_length, padding='post')

# Prepare the input and output
X = english_sequences
y = tulu_sequences

# Prepare the output by shifting
y_input = y[:, :-1]  # Input sequences for the model
y_output = y[:, 1:]  # Output sequences (shifted)

# Before model fitting, print the shapes to debug
print(f'X shape: {X.shape}')  # (num_samples, max_length)
print(f'y_output shape: {y_output.shape}')  # (num_samples, max_length, num_classes)

# Ensure the output sequences are padded to match the model's output shape
y_output = pad_sequences(y_output, maxlen=max_length, padding='post')

# One-hot encoding the output
y_output = tf.keras.utils.to_categorical(y_output, num_classes=len(tulu_tokenizer.word_index) + 1)

# Verify shapes
print(f'X shape: {X.shape}')          # Should be (num_samples, max_length)
print(f'y_output shape: {y_output.shape}')  # Should be (num_samples, max_length, num_classes)

def create_transformer_model(input_dim, output_dim, embedding_dim=128, num_heads=8, ff_dim=128):
    # Input layer
    inputs = tf.keras.Input(shape=(None,))
    
    # Embedding layer
    x = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim)(inputs)
    
    # Transformer layers
    for _ in range(4):
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
        attn_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_output + x)
        
        # Adjust Dense layer to match `embedding_dim`
        x = tf.keras.layers.Dense(ff_dim, activation='relu')(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Output layer
    outputs = tf.keras.layers.Dense(output_dim, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Create the model
model = create_transformer_model(len(english_tokenizer.word_index) + 1, len(tulu_tokenizer.word_index) + 1)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define the learning rate
learning_rate = 1.0  # Adjust as needed

# Compile the model with the custom learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_translation_model.keras', save_best_only=True)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)

# Train the model
history = model.fit(X, y_output, 
                    epochs=100,  # Increased epochs for better training
                    batch_size=64,  # Common batch size
                    validation_split=0.2,  # Use 20% of the data for validation
                    callbacks=[early_stopping, model_checkpoint, lr_scheduler])


# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save the final model
model.save('translation_model.keras')
