# Problem: Apply dropout and regularization (L2) for overfitting control (Titanic Dataset)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

import seaborn as sns
import matplotlib.pyplot as plt

# Load the titanic dataset
titanic = sns.load_dataset('titanic')

# Preprocess the data
# Drop the columns that are not useful for prediction.
titanic = titanic.drop(['embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'sex'], axis=1)

# Fill the missing values.
print(titanic.isnull().sum()) # Only age has missing values
titanic['age'].fillna(titanic['age'].mean(), inplace=True)

X = titanic.drop('survived', axis=1) # Features
y = titanic['survived']

# Split into training and validation dataset.
# 80-20 ratio, training (80%) and validation (20%) datasets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Scaling using Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Build the neural network model with Dropout ad L2 Regularization
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1], ), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5)) # Dropout rate of 50%
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))  # Dropout rate of 30%
model.add(Dense(1, activation='sigmoid'))


# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Convert the history into dataframe for easy viewing.
history_df = pd.DataFrame(history.history)

print(history_df.head())

# Visualize Loss and Accuracy of the Model

# Plot the Loss
plt.figure(figsize=(14, 6))
plt.plot(history_df['loss'], label='Training Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot the Accuracy
plt.figure(figsize=(14, 6))
plt.plot(history_df['accuracy'], label='Training Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()