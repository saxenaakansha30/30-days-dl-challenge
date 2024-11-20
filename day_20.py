# Problem: Build an autoencoder-based anomaly detection system (part 1: data and model setup)
# Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


# Load and preprocess the dataset
data = pd.read_csv('dataset/creditcard.csv')

# Features and target dataset
X = data.drop(columns=['Time', 'Class'])
y = data['Class']

# Standardize Amount column and other features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split training and validation dataset.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Use only non-fraudulent transactions to train auto-encoder
X_train_normal = X_train[y_train == 0]

# Setup the auto-encoder model
model = Sequential()

# Encoder layer
model.add(Dense(14, activation='relu', input_shape=(X_train.shape[1], )))
model.add(Dense(7, activation='relu'))

# Decoder layer
model.add(Dense(14, activation='relu'))
model.add(Dense(X_train.shape[1], activation='linear'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train the model
history = model.fit(
    X_train_normal, X_train_normal,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# Visualize the Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()