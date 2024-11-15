# Problem: Prepare a simple time series dataset (Jena Climate or stock data) for RNN model
# Dataset: https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Load the Jena Climate Dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data = pd.read_csv(url)

# print(data.info())
# print(data.head())

# Inspect and Preprocess the Dataset
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Plot th temp over time to visualize.
plt.figure(figsize=(10, 6))
plt.plot(data, label='Temperature over time')
plt.xlabel('Date')
plt.ylabel('Temp')
plt.title('Daily Minimum Temperature over time')
plt.show()

# Prepare time series data for RNN
def create_time_series_data(data, window_size, future_days=1):
    X = [] # Past observations (of length defined by window_size) used as inputs.
    y = [] # Future value(s) (length defined by target_size) to be predicted.

    for i in range(len(data) - window_size - future_days + 1):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size: i + window_size + future_days])

    return np.array(X), np.array(y)

WINDOW_SIZE = 30 # Use past 30 days data to predict the tenperature

# Convert the Temp column to numpy array.
temperatue_data = data['Temp'].values

# Create time series data
X, y = create_time_series_data(temperatue_data, WINDOW_SIZE)

# Split the data into training(70%), validation(20%) and test_sets(10%)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.3, random_state=42)

# Normalize the data

scalers = MinMaxScaler(feature_range=(0, 1))

# Fit the training, validation and test data
X_train = scalers.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_val = scalers.fit_transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
X_test = scalers.fit_transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

y_train = scalers.transform(y_train)
y_val = scalers.transform(y_val)
y_test = scalers.transform(y_test)

# Create Batches for Model Training

BATCH_SIZE = 32
BUFFER_SIZE = 1000

# Create training tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create validation tensorflow dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create test tensorflow dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(train_dataset)