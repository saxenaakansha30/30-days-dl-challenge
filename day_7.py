# Problem: Fine-tune hyperparameters with Keras Tuner on a small NN

import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import matplotlib.pyplot as plt
import pandas as pd


# Load the dataset
data = load_breast_cancer(as_frame=True)

# Preprocess the data
X = data.data
y = data.target

# Split the dataset into training and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter functions

def build_model(hp):
    model = Sequential()

    # Tune the number of units in the first Dense layer
    hp_units = hp.Int('units', min_value=8, max_value=128, step=8)
    model.add(Dense(units=hp_units, activation='relu', input_shape=(X_train.shape[1], )))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.0001])
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Set Up the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='logs',
    project_name='day_7'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Find the best hyperparameters and build model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best number of units: {best_hps.get('units')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Build the best model with best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the model
history = best_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
)

# Visualize the accuracy and loss of best model
history_df = pd.DataFrame(history.history)

# Plot the loss
plt.figure(figsize=(10, 7))
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], linestyle='--', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Plot the Accuracy
plt.figure(figsize=(10, 7))
plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df['val_accuracy'], linestyle='--', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.show()

# Best val_accuracy So Far: 0.9473684430122375
# Total elapsed time: 00h 00m 08s
# Best number of units: 32
# Best learning rate: 0.01

