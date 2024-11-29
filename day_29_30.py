# Problem: Work on SimCLR self-supervised learning

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10


# Load the CIFAR-10 dataset
(X_train, _), (X_val, _) = cifar10.load_data()

# Combine the training and test data.
# In Self super-vised training technique we dont need label for tranining
X_data = np.concatenate((X_train, X_val), axis=0)

# Normalize the pixel values between -1 and 1 (Helps with `tanh` activation function)
X_data = (X_data.astype('float32') / 127.5) - 1.0

# Define the Augmentation
def data_augment(image):
    # Random crop and resize
    image = tf.image.random_crop(image, size=[28, 28, 3])
    image = tf.image.resize(image, (32, 32))

    # Random flip (Left-Right)
    imagee = tf.image.random_flip_left_right(image)

    # Color distortion
    image = tf.image.random_brightness(image, max_delta=0.5)
    imagee = tf.image.random_contrast(image, lower=0.1, upper=0.9)

    return image

# Visualize some autmented images
fig, axs = plt.subplots(1, 4, figsize=(10, 3))
for i in range(4):
    image = data_augment(X_data[np.random.randint(len(X_data))])
    axs[i].imshow((image + 1) / 2) # Rescale it back to 0 and 1
    axs[i].axis('off')

plt.show()

# Set up the Base Network (Encoder)
def create_encoder():
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    base_model.trainable = True # We want to train the base model from scratch
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, trainable=True)
    x = GlobalAveragePooling2D()(x)

    return tf.keras.Model(inputs, x)

encoder = create_encoder()
encoder.summary()

# Create Project Head
def create_project_head(encoder):
    inputs = encoder.input
    x = encoder.output
    x = Dense(256, activation='relu')(x)
    output = Dense(128)(x) # Final layer will used for contrastive learning

    return tf.keras.Model(inputs, output)

project_head = create_project_head(encoder)
project_head.summary()

# Define the Contrastive Loss
def contrastive_loss(z_i, z_j, temperature=0.5):
    # Normalize the two vectors
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=2)

    # Compute cosine scores
    similarity_matrix = tf.matmul(z_i, z_j, transpose_b=True)
    logits = similarity_matrix / temperature

    # Labels and indices of the positive pair
    batch_size = tf.shape(z_i)[0]
    labels = tf.range(batch_size)

    # Calculate the cross entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    return tf.reduce_mean(loss)


































