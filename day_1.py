# Problem: Predict house prices using a feedforward neural network (NN)

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Step 1: Load the data
data = fetch_california_housing(as_frame=True)
data_df = data.frame

# Step 2: Build the feature and target datasets.
X = data_df.drop('MedHouseVal', axis=1) # Feature Dataset
y = data_df['MedHouseVal'] # Target Dataset

# Step 3: Split the dataset into training and validation datasets.
# We will go with 80-20 ratio, training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the data
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_val)

# Step 5: Build the Neural Network Model
model = Sequential()
model.add( Dense(64, activation='relu', input_shape=(X.shape[1],)) )
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

# Step 6: Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 7: Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Show the learning curve
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation loss across epochs')
plt.show()

# Step 8: Evaluate the model
test_loss = test_mae = model.evaluate(X_val, y_val)
print(f"Validation Mean Absolute Error: {test_mae}")

# Output:
# For Epochs - 50:
# Validation Mean Absolute Error: [0.6920378804206848, 0.6737860441207886]
# For Epochs- 10
# Validation Mean Absolute Error: [0.8164848685264587, 0.745993971824646]
# For Epochs - 6
# Validation Mean Absolute Error: [94.31477355957031, 7.9173150062561035]
# So 50 was a good number.