# Problem: Experiment with optimizers (SGD, Adam, RMSprop) using a pre-built CNN on CIFAR-10

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam, RMSprop


# Load the dataset
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# Normalize pixel values to between 0 and 1
X_train = X_train / 255.0
X_val = X_val / 255.0

# Understand the data.
# X_train_flat = X_train.reshape(-1, 32*32*3)
# X_train_df = pd.DataFrame(X_train_flat)
# print(X_train_df.head())
#
# y_train_df = pd.Series(y_train.flat, name='Label')
# print(y_train_df.head())

# Define the function to build the CNN model
def build_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model

# Train the model with different optimizers.

# List of optimizers
optimizers = {
    'SGD': SGD(learning_rate=0.01, momentum=0.9),
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': RMSprop(learning_rate=0.001)
}


# Dictionary to store the training history
history_dict = {}

for optimizer_name, optimizer in optimizers.items():
    model = build_cnn()

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Training for {optimizer_name}")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    history_dict[optimizer_name] = history.history


# Visualize the loss and accuracy with different optimizers.

# Plot loss.
plt.figure(figsize=(14, 8))
for optimizer_name in optimizers:
    plt.plot(history_dict[optimizer_name]['loss'], label=f'Training Loss for ({optimizer_name})')
    plt.plot(history_dict[optimizer_name]['val_loss'], label=f'Validation Loss for ({optimizer_name})')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss with Different Optimizers')
plt.legend()
plt.show()

# Plot Accuracy.
plt.figure(figsize=(14, 8))
for optimizer_name in optimizers:
    plt.plot(history_dict[optimizer_name]['accuracy'], label=f'Training Accuracy for ({optimizer_name})')
    plt.plot(history_dict[optimizer_name]['val_accuracy'], label=f'Validation Accuracy for ({optimizer_name})')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy with Different Optimizers')
plt.legend()
plt.show()