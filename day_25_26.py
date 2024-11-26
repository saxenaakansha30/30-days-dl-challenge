# Problem: Implement CycleGAN for style transfer (e.g., horse to zebra conversion)
# Dataset: https://www.kaggle.com/datasets/balraj98/horse2zebra-dataset/data?select=trainA

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

# Set up the directories to save the generted images
if not os.path.exists('generated_images'):
    os.makedirs("generated_images")

# Define the Hyper-parameters
EPOCHS = 100
BATCH_SIZE = 1
SAVE_INTERVAL = 10

# Label for real and fake images for training discriminator
REAL_LABEL = np.ones((BATCH_SIZE, 16, 16, 1)) # Will be of shape 16x16 with 1 channel
FAKE_LABEL = np.zeros((BATCH_SIZE, 16, 16, 1))

for epoch in range(EPOCHS):
    for real_a, real_b in Dataset.zip((train_horses, train_zebras)).take(100):
        # Generate fake images using the generator
        fake_a = generator_g.predict(real_a)
        fake_b = generator_f.predict(real_b)

        # Train discriminator with real and fake images
        # Train Discriminator A
        d_a_real_loss = discriminator_a.train_on_batch(real_a, REAL_LABEL)
        d_a_fake_loss = discriminator_a.train_on_batch(fake_a, FAKE_LABEL)
        d_a_loss = 0.5 * np.add(d_a_real_loss, d_a_fake_loss)

        # Train Discriminator B
        d_b_real_loss = discriminator_b.train_on_batch(real_b, REAL_LABEL)
        d_b_fake_loss = discriminator_b.train_on_batch(fake_b, FAKE_LABEL)
        d_b_loss = 0.5 * np.add(d_b_real_loss, d_b_fake_loss)

        # Train generator to fool discriminator and maintain cycle consistency
        g_loss = combined_model.train_on_batch([real_a, real_b], [REAL_LABEL, REAL_LABEL, real_a, real_b, real_a, real_b])

    # Print the progress
    print(f"Epoch: {epoch + 1} / {EPOCHS}")
    print(f"D_A_Loss: {d_a_loss[0]:.4f}, D_B_LOSS: {d_b_loss[0]:.4f}")
    print(f"G_loss: {g_loss}")

    # Save generated images at regular interval.
    if (epoch + 1) % SAVE_INTERVAL == 0:
        fake_a = generator_g.predict(real_a)
        fake_b = generator_f.predict(real_b)

        # Visualize the generated images
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.title("Original Horse")
        plt.imshow((real_a[0] + 1) / 2) # Rescale to [0,1]
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("Generated Zebra")
        plt.imshow((fake_b[0] + 1) / 2) # Rescale to [0,1]
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title("Original Zebra")
        plt.imshow((real_b[0] + 1) / 2) # Rescale to [0,1]
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title("Generated Horse")
        plt.imshow((fake_a[0] + 1) / 2) # Rescale to [0,1]
        plt.axis('off')

        plt.savefig(f"generated_images/epochs_{epoch + 1}.png")

        # plt.show()






