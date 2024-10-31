# Problem: Classify handwritten digits using a simple NN on MNIST

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Load the data
(X_train, y_train), (X_val, y_val) = mnist.load_data()

# Step 3: Prepare and Preprocess the data
# Neural Networks Expect 1D Vectors
# So we flatten the 28*28 pixel image into 784 1D vector
X_train_flat = X_train.reshape(-1, 28*28)
X_val_flat = X_val.reshape(-1, 28*28)

# Convert the X_train and y_train into dataframes for better visualization.
X_train_df = pd.DataFrame(X_train_flat)
y_train_df = pd.Series(y_train, name='Label')

# Step 4: Build the Neural Network
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784, )))
model.add(Dense(64, activation='relu'))
# The output layer has 10 neurons, each representing one of the digits 0-9.
# softmax activation is used to turn the output into probabilities, with the sum equal to 1.
model.add(Dense(10, activation='softmax'))

# Step 5: Compile the model
model.compile(
    optimizer='adam',
    loss=['sparse_categorical_crossentropy'],
    metrics=['accuracy']
)

# Step 6: Train the model
history = model.fit(X_train_flat, y_train, validation_data=(X_val_flat, y_val), epochs=10, batch_size=32, verbose=1)

# Step 7: Evaluate the Model
val_loss, val_accuracy  = model.evaluate(X_val_flat, y_val)
print(f"Validation data Accuracy: {val_accuracy}")

# Output:
# Validation data Accuracy: 0.968999981880188

# Step 8: Visualize the Training and Validation Performance
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Step 9: Make some prediction
predictions = model.predict(X_val_flat[:5])

for i in range(5):
    plt.imshow(X_val[i].reshape(28, 28), cmap='gray')
    plt.title(f"Prediction {np.argmax(predictions[i])}, True: {y_val[i]}")
    plt.axis('off')
    plt.show()

