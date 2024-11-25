# Problem: Implement CycleGAN for style transfer (e.g., horse to zebra conversion)

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.data import Dataset

# Load the dataset
HORSE_DIR = 'dataset/horse2zebra/trainA/'
ZEBRA_DIR = 'dataset/horse2zebra/trainB/'

# Helper function to load images from directories
def load_images_from_directory(directory, size=(128, 128)):
    images = []

    for filepath in glob(os.path.join(directory, '*.jpg')):
        image = load_img(filepath, target_size=size)
        image = img_to_array(image)
        images.append(image)

    return np.array(images)

# Load horses and zebra images
horse_images = load_images_from_directory(HORSE_DIR)
zebra_images = load_images_from_directory(ZEBRA_DIR)


# Normalize it between [-1, 1]
horse_images = (horse_images - 127.5) / 127.5
zebra_images = (zebra_images - 127.5) / 127.5

# Convert to tensorflow dataset and batch them
train_horses = Dataset.from_tensor_slices(horse_images).batch(1)
train_zebras = Dataset.from_tensor_slices(zebra_images).batch(1)


# Build the Generator Model
def build_generator():
    inputs = Input(shape=(128, 128, 3))

    # Encoder: Downsampling layers
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Decoder: Upsampling layers
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)

    return Model(inputs, x)

# Build generator for both transformations
generator_g = build_generator() # Horse to zebra
generator_f = build_generator() # Zebra to horse

generator_g.summary()
generator_f.summary()

# Build the discriminator Model

# Define the discriminator model
def build_discriminator():
    inputs = Input(shape=(128, 128, 3))

    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1, kernel_size=4, padding='same')(x)

    return Model(inputs, x)

# Build the dicriminator for both domains
discriminator_a = build_discriminator() # For Domain A (Horses)
discriminator_b = build_discriminator() # For Domain B (Zebras)

discriminator_a.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='mse',
    metrics=['accuracy']
)

discriminator_b.compile(
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    loss='mse',
    metrics=['accuracy']
)

discriminator_a.summary()
discriminator_b.summary()

# Build Cycle GAN Model

# Define the combined cycle GAN model
def build_combined(generator_g, generator_f, discriminator_a, discriminator_b):

    discriminator_a.trainable = False
    discriminator_b.trainable = False

    # Real input images for both the domain
    input_a = Input(shape=(128, 128, 3)) # Horses
    input_b = Input(shape=(128, 128, 3)) # Zebras

    # Forward cycle: A -> B -> A
    fake_b = generator_g(input_a)
    cycle_a = generator_f(fake_b)

    # Backward cycle: B -> A -> B
    fake_a = generator_f(input_b)
    cycle_b = generator_g(fake_a)

    # Identifying mapping preserving original features
    same_a = generator_f(input_a)
    same_b = generator_g(input_b)

    # Discriminators for the generated images
    valid_a = discriminator_a(fake_a)
    valid_b = discriminator_b(fake_b)

    # Define the combined model
    model = Model(
        inputs=[input_a, input_b],
        outputs=[
            valid_a,
            valid_b,
            cycle_a,
            cycle_b,
            same_a,
            same_b
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss=['mse', 'mse', 'mse', 'mse', 'mse', 'mse'],
        loss_weights=[1, 1, 10, 10, 5, 5]
    )

    return  model

combined_model = build_combined(generator_g, generator_f, discriminator_a, discriminator_b)
combined_model.summary()









