# Problem: Compare activation functions (ReLU, Sigmoid, Tanh) on Fashion MNIST

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt


# Load the dataset
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# Normalize the data to the range [0, 1]
X_train = X_train / 255.0
X_val = X_val / 255.0


def build_model(activation_function):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation=activation_function))
    model.add(Dense(64, activation=activation_function))
    model.add(Dense(10, activation='softmax'))

    return model

# Compile and Train the model on different activation functions.
activation_fucntions = ['relu', 'sigmoid', 'tanh']

# Dictionary to store the history of training on differnet activation functions.
history_dict = {}

for activation in activation_fucntions:
    model = build_model(activation)

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Training for actiation function: {activation}")
    # Train the model.
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    history_dict[activation] = history.history


# Visualize the accuracy and loss.

# Plot Training and Validation loss for each activation function.
plt.figure(figsize=(14, 8))
for activation in activation_fucntions:
    plt.plot(history_dict[activation]['loss'], label=f'Training Loss ({activation})')
    plt.plot(history_dict[activation]['val_loss'],  linestyle='--' , label=f'Validation Loss ({activation})')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss for each Activation Function')

# Plot Training nad Validation Accuracy for each activation function.
plt.figure(figsize=(14, 8))
for activation in activation_fucntions:
    plt.plot(history_dict[activation]['accuracy'], label=f'Training Accuracy ({activation})')
    plt.plot(history_dict[activation]['val_accuracy'],  linestyle='--' ,label=f'Validation Accuracy ({activation})')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy for each Activation Function')

plt.show()