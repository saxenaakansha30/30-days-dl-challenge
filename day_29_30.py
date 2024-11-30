# Problem: Work on SimCLR self-supervised learning

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# Load the CIFAR-10 dataset
(X_train, y_train), (X_val, y_val) = cifar10.load_data()

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
    x = base_model(inputs, training=True)
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
    z_j = tf.math.l2_normalize(z_j, axis=1)

    # Compute cosine scores
    similarity_matrix = tf.matmul(z_i, z_j, transpose_b=True)
    logits = similarity_matrix / temperature

    # Labels and indices of the positive pair
    batch_size = tf.shape(z_i)[0]
    labels = tf.range(batch_size)

    # Calculate the cross entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    return tf.reduce_mean(loss)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128
BUFFER_SIZE = 10000

def prepare_data(x_data):

    dataset = tf.data.Dataset.from_tensor_slices(x_data)
    dataset = dataset.shuffle(BUFFER_SIZE).map(data_augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset

dataset = prepare_data(X_data)
optimizer = Adam(learning_rate=0.0003)

# Training Steps
@tf.function
def train_steps(batch):
    # Generate two augmented views of the image
    augmented_1 = tf.map_fn(data_augment, batch)
    augmented_2 = tf.map_fn(data_augment, batch)

    with tf.GradientTape() as tape:
        # Encoder and project both
        z_i = project_head(augmented_1, training=True)
        z_j = project_head(augmented_2, training=True)

        # Calculate the contrastive loss
        loss = contrastive_loss(z_i, z_j)

    # Apply the gradient
    gradients = tape.gradient(loss, project_head.trainable_variables)
    optimizer.apply_gradients(zip(gradients, project_head.trainable_variables))

    return loss

# Train the model


EPOCHS = 10
for epoch in range(EPOCHS):
    epoch_loss_avg = tf.keras.metrics.Mean()
    for batch in dataset:
        loss = train_steps(batch)
        epoch_loss_avg.update_state(loss)

    print(f"Epoch: {epoch + 1}, loss: {epoch_loss_avg.result().numpy()}")

# Evaluate the Model
encoder.trainable = False

# Create a simple classifier
classifier = tf.keras.Sequential(
    encoder,
    Dense(10, activation='softmax')
)

classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Split into training (80%) and validation (20%) dataset
X_train, X_val = train_test_split(X_data, test_size=0.2, random_state=42)

# Train the classifier
classifier.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=128
)





















