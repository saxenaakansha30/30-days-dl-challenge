# Problem: Attention Mechanism for LSTM in Machine Translation

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dot, LSTM, Dense, Embedding, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K


# Data preprocessing
max_encoder_seq_length = 20
max_decoder_seq_length = 20
input_vocab_size = 10000
output_vocab_size = 10000
embedding_dim = 128

# Set filters to empty string to include special tokens
input_tokenizer = Tokenizer(num_words=input_vocab_size, filters='')
output_tokenizer = Tokenizer(num_words=output_vocab_size, filters='')

input_sequences = ["I am learning deep learning.", "This is a test sentence."]
output_sequences = ["<start> Je suis en train d'apprendre l'apprentissage profond. <end>",
                    "<start> Ceci est une phrase de test. <end>"]

# Fit tokenizers
input_tokenizer.fit_on_texts(input_sequences)
output_tokenizer.fit_on_texts(output_sequences)

# Tokenize and pad sequences
input_sequences = input_tokenizer.texts_to_sequences(input_sequences)
output_sequences = output_tokenizer.texts_to_sequences(output_sequences)

input_sequences = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_decoder_seq_length, padding='post')

# Retrieve the start token from the tokenizer
try:
    start_token = output_tokenizer.word_index['<start>']
except KeyError:
    raise ValueError(
        "'<start>' token not found in the tokenizer. Make sure your output sequences contain '<start>' token.")

# Build decoder inputs
decoder_inputs = np.full((output_sequences.shape[0], 1), start_token)  # Shape: (batch_size, 1)
decoder_inputs = np.concatenate([decoder_inputs, output_sequences[:, :-1]], axis=1)

# Build an encoder
encoder_inputs = Input(shape=(max_encoder_seq_length,))
encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(128, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

# Build the decoder
decoder_inputs_layer = Input(shape=(max_decoder_seq_length,))
decoder_embedding = Embedding(input_dim=output_vocab_size, output_dim=embedding_dim)(decoder_inputs_layer)
decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])

# Attention Mechanism
attention = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, encoder_outputs])

decoder_combined_context = Concatenate(axis=-1)([context, decoder_outputs])

decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_output_final = decoder_dense(decoder_combined_context)

# Define the training Model
model = Model([encoder_inputs, decoder_inputs_layer], decoder_output_final)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train the model
model.fit(
    [input_sequences, decoder_inputs],
    output_sequences,
    batch_size=64,
    epochs=10,
    validation_split=0.2
)

# Define Encoder Model for Inference
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Define Decoder Model for Inference
decoder_state_input_h = Input(shape=(128,))
decoder_state_input_c = Input(shape=(128,))
encoder_output_input = Input(shape=(max_encoder_seq_length, 128))

decoder_embedding2 = Embedding(input_dim=output_vocab_size, output_dim=embedding_dim)(decoder_inputs_layer)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_embedding2,
                                                    initial_state=[decoder_state_input_h, decoder_state_input_c])

attention2 = Dot(axes=[2, 2])([decoder_outputs2, encoder_output_input])
attention2 = Activation('softmax')(attention2)
context2 = Dot(axes=[2, 1])([attention2, encoder_output_input])

decoder_combined_context2 = Concatenate(axis=-1)([context2, decoder_outputs2])
decoder_output_final2 = decoder_dense(decoder_combined_context2)

decoder_model = Model(
    [decoder_inputs_layer, encoder_output_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_output_final2, state_h2, state_c2]
)
