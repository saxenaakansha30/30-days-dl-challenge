# Problem: Build a simple CNN for CIFAR-10 image classification

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd


# Load the dataset
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# Preprocess the data

# Normalize the pixel values between 0 and 1
X_train = X_train / 255.0
X_val = X_val / 255.0

# Understand the data.
# X_train_flat = X_train.reshape(-1, 32*32*3)
# X_train_df = pd.DataFrame(X_train_flat)
# print(X_train_df.head())
#
# y_train_df = pd.Series(y_train.flat, name='Label')
# print(y_train_df.head())

# Define the CNN model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))

# Add second Convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model.
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate the performance of the model.
test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=2)
print(f'The Loss is {test_loss}, and accuracy is {test_accuracy}')

# Visualize the performance
history_df = pd.DataFrame(history.history)

plt.figure(figsize=(10, 7))
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Training and Validation Loss')
plt.title('Training and Validation Loss of CNN')
plt.legend()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Training and Validation Accuracy')
plt.title('Training and Validation Accuracy of CNN')
plt.legend()
plt.show()

# Output
# The Loss is 0.8811351656913757, and accuracy is 0.6985999941825867