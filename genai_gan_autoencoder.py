import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and preprocess MNIST data
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# Step 2: Autoencoder
# Encoder
encoder = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu')
])
# Decoder
decoder = models.Sequential([
    layers.Input(shape=(64,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])
# Autoencoder
autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

print("\nTraining Autoencoder...")
autoencoder.fit(X_train, X_train, epochs=20, batch_size=256, validation_data=(X_test, X_test), verbose=2)

# Evaluate Autoencoder
reconstructed_images = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructed_images))
print(f'\nAutoencoder Reconstruction MSE: {mse:.6f}')

# Step 3: GAN
# Generator
def build_generator():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(784, activation='sigmoid')
    ])
    return model
# Discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = models.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

epochs = 3000  # Reduce for faster demo; increase for better results
batch_size = 64
half_batch = batch_size // 2

print("\nTraining GAN...")
for epoch in range(epochs):
    # Train Discriminator
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_images = X_train[idx]
    real_labels = np.ones((half_batch, 1))
    noise = np.random.normal(0, 1, (half_batch, 100))
    fake_images = generator.predict(noise, verbose=0)
    fake_labels = np.zeros((half_batch, 1))
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    gan_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, gan_labels)
    # Print and visualize
    if epoch % 500 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss_real[0]:.4f}, Acc: {d_loss_real[1]:.4f}, Generator Loss: {g_loss:.4f}")
        gen_imgs = generator.predict(np.random.normal(0, 1, (10, 100)), verbose=0)
        plt.figure(figsize=(10,2))
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(gen_imgs[i].reshape(28,28), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Generated Images at Epoch {epoch}')
        plt.show()

print("\nGAN training complete. Inspect the generated images above for quality.") 