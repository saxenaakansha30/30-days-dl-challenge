# Problem: Conditional GAN (CGAN) for Generating Specific Images from Fashion MNIST

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D, \
    Input, Concatenate
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the Fashion MNIST dataset
(X_train, y_train), (_, _) = fashion_mnist.load_data()

# Normalize the images to the range [-1, 1] to fit the tanh activation function in the generator
X_train = (X_train - 127.5) / 127.5
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

# One-hot encode the labels for conditioning
num_classes = 10
y_train = to_categorical(y_train, num_classes)


# Function to build the generator
def build_generator():
    # Inputs for the generator
    noise_input = Input(shape=(100,))
    label_input = Input(shape=(num_classes,))

    # Concatenate noise and label to create the input for the generator
    model_input = Concatenate()([noise_input, label_input])

    x = Dense(7 * 7 * 256, activation='relu')(model_input)
    x = Reshape((7, 7, 256))(x)
    x = BatchNormalization(momentum=0.8)(x)

    # Upsample to 14x14
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    # Upsample to 28x28
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    # Final layer to generate an image with 28x28 dimensions and 1 channel
    img_output = Conv2D(1, kernel_size=7, activation='tanh', padding='same')(x)

    return Model([noise_input, label_input], img_output)


# Build the generator model
generator = build_generator()
generator.summary()


# Function to build the discriminator
def build_discriminator():
    # Inputs for the image and the label
    img_input = Input(shape=(28, 28, 1))
    label_input = Input(shape=(num_classes,))

    # Embed the label and reshape to match the image shape
    label_embedding = Dense(28 * 28)(label_input)
    label_embedding = Reshape((28, 28, 1))(label_embedding)

    # Concatenate the image and label embedding
    combined_input = Concatenate()([img_input, label_embedding])

    # Flatten the combined input and pass through dense layers
    x = Flatten()(combined_input)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Final output layer to classify real (1) or fake (0)
    validity_output = Dense(1, activation='sigmoid')(x)

    # Create the model that takes the image and label as input
    return Model([img_input, label_input], validity_output)


# Build and compile the discriminator model
discriminator = build_discriminator()
discriminator.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
discriminator.summary()

# Build the combined CGAN model
# Freeze the discriminator's layers when training the combined CGAN model
discriminator.trainable = False

# Inputs for noise and label
noise_input = Input(shape=(100,))
label_input = Input(shape=(num_classes,))

# Generate an image from the noise and label input
img = generator([noise_input, label_input])

# Use the discriminator to classify the generated image with the label
validity = discriminator([img, label_input])

# Define the combined CGAN model
cgan = Model([noise_input, label_input], validity)
cgan.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy')
cgan.summary()

# Training the CGAN

# Training Parameters
epochs = 10000
batch_size = 32
save_interval = 1000

# Labels for real and fake images
real = np.ones((batch_size, 1)) * 0.9  # Smoothed label for real images
fake = np.zeros((batch_size, 1)) + 0.1  # Noisy label for fake images

for epoch in range(epochs):

    # Train the discriminator with real images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    real_labels = y_train[idx]

    # Generate fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_labels = np.eye(num_classes)[np.random.choice(num_classes, batch_size)]
    gen_imgs = generator.predict([noise, fake_labels])

    # Train the discriminator on real and fake images
    d_loss_real = discriminator.train_on_batch([real_imgs, real_labels], real)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, fake_labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator via the combined CGAN model
    noise = np.random.normal(0, 1, (batch_size, 100))
    sampled_labels = np.eye(num_classes)[np.random.choice(num_classes, batch_size)]
    g_loss = cgan.train_on_batch([noise, sampled_labels], real)

    # Display training progress and save images at intervals
    if epoch % save_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

        # Save generated images to visualize training progress
        generated_imgs = generator.predict([noise, sampled_labels])
        generated_imgs = 0.5 * generated_imgs + 0.5  # Rescale from [-1, 1] to [0, 1]

        plt.figure(figsize=(5, 5))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(generated_imgs[i, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.show()
