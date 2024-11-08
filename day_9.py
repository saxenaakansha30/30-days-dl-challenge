# Problem: Modify CNN with pooling layers and visualize filters
# dataset: cirfar-100
import pylab as pl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load the dataset
(X_train, y_train), (X_val, y_val) = cifar100.load_data()

# X_train_flat = X_train.reshape(-1, 32*32*3)
# X_train_df = pd.DataFrame(X_train_flat)
# print(X_train_df.head())

# Normalize the data from 0-255 to 0-1
X_train = X_train / 255.0
X_val = X_val / 255.0

# print(y_train.shape)
# print(y_val.shape)
# y_train_flat = y_train.flat
# y_train_df = pd.Series(y_train_flat)
# print(y_train_df.head())

# Build the  model
model = Sequential()

# First convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(MaxPooling2D(2, 2))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(100, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Visualize the Performance
history_df = pd.DataFrame(history.history)

# Plot the Loss
plt.figure(figsize=(10, 7))
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot the Accuracy
plt.figure(figsize=(10, 7))
plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Visualize the filters of the first convolutional layer
first_layer = model.layers[0]
print(first_layer)
filters, biases = first_layer.get_weights()
print(filters)
print(biases)

# Normalize the filter values to 0-1 for visualization
# 1, 255
# 34 = 34 - 1 / 255 = 33 / 255 = 0.12
# 255 = 255 - 1 / 255 = 254 / 255
# min-max normalization
filters_min, filters_max = filters.min(), filters.max()
filters = (filters - filters_min) / (filters_max - filters_min)

# Plot the filters
number_of_filters = 6
fig, axes = plt.subplots(1, number_of_filters, figsize=(20, 5))

for i in range(number_of_filters):
    # Get the filters
    #  selects the i-th filter.
    f = filters[:, :, 0, i]  # Only plot the first channel for simplicity

    ax = axes[i]
    ax.imshow(f, cmap='viridis')  # Plot the filter
    ax.axis('off')

plt.suptitle('Filters of the first convolutional layers')
plt.show()

# The colorful squares that you see in the first three filters are the visual representation of the weights in the filter.
# These patterns represent how the filter is "looking" at the input images — the colors correspond to the weights of the
# filter at different positions, where bright colors represent high weights and dark colors represent low weights.