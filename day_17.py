# Problem: Build an LSTM model for sentiment analysis (IMDb Dataset)

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM
import matplotlib.pyplot as plt


# Load and preprocess the IMDB dataset
vocab_size = 10000
(X_train, y_train), (X_val, y_val) = imdb.load_data(num_words=vocab_size) # Load imdb data with only 10,000 common words

# Padding sequences to ensure uniform length of review.
max_length = 200
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_val = pad_sequences(X_val, maxlen=max_length, padding='post')

# Define the LSTM Model
model = Sequential()

embedding_dimension = 128
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dimension,
        input_length=max_length
    )
)

model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

# Visualization: Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Traning and Validation Loss Of LSTM Model')
plt.legend()
plt.show()







