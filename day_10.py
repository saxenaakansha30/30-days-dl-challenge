# Problem: Use pre-built data augmentation methods in Keras on Fashion MNIST


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd


# Load the data
(X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()

# Reshape the data to add the channel dimension (grayscale has 1 channel)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape((X_val.shape[0], 28, 28, 1))

# Visualise the dataset
# X_train_df = pd.DataFrame(X_train)
# y_train_df = pd.Series(y_train)
# print(y_train_df.head())

# Normalize the pixel values between 0-1
X_train = X_train / 255.0
X_val = X_val / 255.0


# Apply Data Augmentation Using ImageGenerator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# Visualize some augmented images.
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
    # Create a grid 3*3 size
    fig, ax = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        ax[i//3, i%3].imshow(X_batch[i].reshape(28, 28), cmap='gray')

    plt.suptitle('Augmented Images')
    plt.show()
    break

# Build the model
model = Sequential()

# Add layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model with Augmented Data
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=9),
    validation_data=(X_val, y_val),
    epochs=10,
    verbose=1
)

# Visualize the performance
history_df = pd.DataFrame(history.history)

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the accuracy
plt.figure(figsize=(10, 6))
plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()