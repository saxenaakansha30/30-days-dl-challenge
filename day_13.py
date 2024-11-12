# Problem: Explore image segmentation with U-Net for a small portion of Carvana dataset

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
from PIL import Image

# Set the path to the dataset
IMAGE_DIR = "dataset/carvana/train/"
MASK_DIR = "dataset/carvana/train_masks/"

def load_data(image_dir, mask_dir, image_size=(128, 128)):
    images = []
    masks = []

    images_to_load = 100 # Change it to any number based on your sample size.
    # Load the images and masks.
    for image_name in os.listdir(image_dir):
        # Skip the hidden files.
        if image_name.startswith("."):
            continue

        # Construct path to image and corresponding masks
        image_path = os.path.join(image_dir, image_name)

        # Mask file name path
        base_name = image_name.replace(".jpg", "")
        mask_name = f"{base_name}_mask.gif"
        mask_path = os.path.join(mask_dir, mask_name)

        # Load the image using open-cv
        image = cv2.imread(image_path)
        if image is None:
            print(f"{image_name} is not found at {image_path}")
            continue
        else:
            # Resize and Normalize
            image = cv2.resize(image, image_size) / 255.0

        # Load the mask using PIL (Pillow)
        try:
            mask = Image.open(mask_path)
            mask = mask.convert('L') # Convert to grayscale
            mask = np.array(mask) # Convert to numpy array
            mask = cv2.resize(mask, image_size) # Resize to same image size
            mask = mask / 255.0 # Normalize
            mask = np.expand_dims(mask, axis=-1) # Add channel dimension
        except Exception as e:
            print(f"{mask_path} could not be loaded because of error: {e}")
            continue

        images.append(image)
        masks.append(mask)

        images_to_load = images_to_load - 1
        if images_to_load == 0:
            break

    return np.array(images), np.array(masks)

# Load the data
X, y = load_data(IMAGE_DIR, MASK_DIR)
print("Dataset loaded successfully")
print(f"Shape of images {X.shape} and masks {y.shape}")
# Shape of images (200, 128, 128, 3) and masks (200, 128, 128, 1)

# Plot a few images and their corresponding masks
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.imshow(X[i])
    plt.title("Car image")
    plt.axis('off')

    plt.subplot(2, 3, i + 4)
    plt.imshow(y[i].squeeze(), cmap='gray')
    plt.title("Mask Image")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Build the U-Net Model
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)

    # Contracting Path (Encoder)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    # Two rounds of feature finding to ensure the model captures enough details before moving on to the next level of analysis.
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck: The bottleneck is a special part of the U-Net architecture.
    # It’s the deepest part of the network, right in the middle where the contracting path ends,
    # and the expanding path begins.
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Expanding Path (Decoding)
    u5 = UpSampling2D((2, 2))(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model

# Instantiate the model
model = unet_model()
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X,
    y,
    validation_split=0.1,
    epochs=10,
    batch_size=8
)

# Evaluate the model and visualize the Results
sample_image = X[0]
sample_mask = y[0]

# Expand dimensions to make it compatible with the model input
sample_image_expanded = np.expand_dims(sample_image, axis=0)

# Predict the mask
predicted_mask = model.predict(sample_image_expanded)[0]

# Plot Original image, true mask and predicted mask.
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(sample_image)

plt.subplot(1, 3, 2)
plt.title("True Mask")
plt.imshow(sample_mask.squeeze(), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(predicted_mask.squeeze(), cmap='gray')

plt.show()

























