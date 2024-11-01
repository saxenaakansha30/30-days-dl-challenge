# Problem: Explore backpropagation theory and tweak learning rates (MNIST)

import numpy as np
import pandas as pd
import tensorflow.keras.optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Load the data
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# Normalize the data to range 0-1
X_train = X_train / 255.0
X_val = X_val / 255.0

# Flatten the 28*28 pixel image to 784  1D vector.
# X_train_flat = X_train.reshape(-1, 28*28)
#
# X_train_df = pd.DataFrame(X_train_flat)
# y_train_df = pd.Series(y_train, name='Label')
#
# print(X_train_df.head())
# print(y_train_df.head())

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Experiment with learning rate
learning_rates = [0.01, 0.001, 0.0001]

# Store the training history for each learning rate
history_lr = {}

for lr in learning_rates:
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    print(f"Training with learning rate: {lr}")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

    # Save the training info in the history_lr dictionary
    history_lr[lr] = history.history


# Visualize the training and validation loss for learning rates
plt.figure(figsize=(14, 8))

for lr in learning_rates:
    plt.plot(history_lr[lr]['loss'], label=f'Training loss (lr={lr})')
    plt.plot(history_lr[lr]['val_loss'], label=f'Validation Loss (lr={lr}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for different learning rates')
plt.legend()
plt.show()

# Visualize the training and validation accuracy for learning rates
plt.figure(figsize=(14, 8))

for lr in learning_rates:
    plt.plot(history_lr[lr]['accuracy'], label=f'Training accuracy (lr={lr})')
    plt.plot(history_lr[lr]['val_accuracy'], label=f'Validation accuracy (lr={lr}')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy for different learning rates')
plt.legend()
plt.show()

# The learning rate of 0.001 seems to be the best option among the three. It has the following advantages:
#
# Smooth convergence: The training and validation loss both decrease steadily, without fluctuations, which shows a stable learning process.
# High accuracy: The training and validation accuracy both improve steadily, indicating effective learning and good generalization.
# Learning rate = 0.01 is too high, leading to fluctuations and instability, possibly due to the model overshooting the optimal point.
#
# Learning rate = 0.0001 is too low, resulting in slower convergence and the model not effectively learning within the given number of epochs.