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
from sklearn.metrics import classification_report, confusion_matrix


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
model.add(Dense(20, activation='relu', input_shape=(X_train.shape[1], )))
model.add(Dense(10, activation='relu'))

# Latent representation to capture the kep patterns
model.add(Dense(5, activation='relu'))

# Decoder layer
model.add(Dense(10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(X_train.shape[1], activation='linear'))  # Reconstruction of the original input

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse'
)

# Train the model
history = model.fit(
    X_train_normal, X_train_normal,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Calculate reconstruction error
X_val_pred = model.predict(X_val)
# print(X_val_pred)

# Calculate mean square error for each input sample
reconstruction_errors = np.mean(np.power((X_val - X_val_pred), 2), axis=1)

print(reconstruction_errors)

threshold = np.percentile(reconstruction_errors[y_val == 0], 95)
print(f"Threshold for anomaly detection is: {threshold}")

y_prediction = [1 if error > threshold else 0 for error in reconstruction_errors]

# Actual Labels for evaluation
print(f"Actual Anomalies: {sum(y_val)}, Detected: {sum(y_prediction)}")

# Evaluate the model's performance
confusion_matrix = confusion_matrix(y_val, y_prediction)
print(f"Confusion Matrix:\n {confusion_matrix}")

classification_report = classification_report(y_val, y_prediction)
print(f"Classification Report:\n {classification_report}")

# Visualize the Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()