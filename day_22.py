# Problem: GAN Basics: Understand GAN architecture and set up the framework (on MNIST or Fashion MNIST)


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import pandas as pd
import os

# Load the data
(X_train, _), (_, _) = fashion_mnist.load_data()

# X_train_flat = X_train.reshape(-1, 28*28)
# df = pd.DataFrame(X_train_flat)

#Normalize between -1 and 1 as it helps tanh activation function.
X_train = (X_train - 127.5) / 127.5
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

# Build Generator
# Build random noise and generate images of same shape as of input fashion mnist data (28x28)
def build_generator():
    model = Sequential()

    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(28*28*1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))

    return  model

generator = build_generator()
generator.summary()

# Build the discriminator
def build_discriminator():
    model = Sequential()

    model.add(Flatten(input_shape=(28, 28, 1)))

    # Add layers to process flatten image
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))

    # Final output layer to classify real(1) or fake(0)
    model.add(Dense(1, activation='sigmoid'))

    return model

discriminator = build_discriminator()

# Compile the model
discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
discriminator.summary()

# Build GAN
discriminator.trainable = False

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    return model

gan = build_gan(generator, discriminator)
gan.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Training the GAN

# Training Parameters
epochs = 10000
batch_size = 64
save_intervals = 1000

# Label for real and fake images
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):

    # Train the discriminator
    random_idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[random_idx]

    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_images, real)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the Generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    gan_loss = gan.train_on_batch(noise, real)

    if epoch % save_intervals == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {gan_loss}]")

        # Save the generated images to visualize training progress
        generated_images = generator.predict(noise)
        generated_images = 0.5 * generated_images + 0.5 # Rescale from -1 and 1 to 0 and 1

        plt.figure(figsize=(5, 5))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(generated_images[i, :, :, 0], cmap='gray')
            plt.axis('off')

        plt.show()







