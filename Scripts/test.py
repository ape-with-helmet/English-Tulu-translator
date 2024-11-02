import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time

# Load the best model
model = tf.keras.models.load_model('best_translation_model.keras')

# Load the tokenizers
with open('english_tokenizer.pkl', 'rb') as file:
    english_tokenizer = pickle.load(file)

with open('tulu_tokenizer.pkl', 'rb') as file:
    tulu_tokenizer = pickle.load(file)

# Function to preprocess input sentence
def preprocess_input(input_sentence):
    # Convert the input sentence to a sequence
    sequence = english_tokenizer.texts_to_sequences([input_sentence])
    
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=20, padding='post')  # Adjust maxlen if necessary
    
    print(f'Input Sentence: {input_sentence}')
    print(f'Preprocessed Input: {padded_sequence}')
    
    return padded_sequence

# Function to decode the predicted output
def decode_sequence(sequence):
    words = []
    for idx in sequence:
        if idx != 0:  # Skip padding (index 0)
            words.append(tulu_tokenizer.index_word.get(idx, ''))  # Use .get to avoid KeyError
    return ' '.join(words)

# Function to translate input sentence
def translate_sentence(input_sentence):
    # Preprocess the input
    padded_input = preprocess_input(input_sentence)

    # Make predictions
    predicted_probs = model.predict(padded_input)
    predicted_indices = np.argmax(predicted_probs, axis=-1)[0]  # Get the first prediction
    
    print(f'Predicted Indices: {predicted_indices}')
    
    # Decode the predicted sequence
    predicted_translation = decode_sequence(predicted_indices)
    
    return predicted_translation

# Function to evaluate model performance on multiple sentences
def evaluate_model(sentences):
    results = {}
    for sentence in sentences:
        start_time = time.time()  # Start time for translation
        translation = translate_sentence(sentence)
        end_time = time.time()  # End time for translation
        
        results[sentence] = {
            'translation': translation,
            'time_taken': end_time - start_time
        }
        
    return results

# Example usage
if __name__ == "__main__":
    # List of sentences to test the translation model
    test_sentences = [
        "The change was written by Chandu",
        "I love to learn new languages",
        "What time is the meeting?",
        "He is playing football in the park",
        "Please send me the report by tomorrow"
    ]
    
    # Evaluate the model
    results = evaluate_model(test_sentences)
    
    # Display the results
    for sentence, result in results.items():
        print(f'Original: {sentence}')
        print(f'Translation: {result["translation"]}')
        print(f'Time Taken: {result["time_taken"]:.4f} seconds\n')
