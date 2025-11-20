import tensorflow as tf
import numpy as np
import jax.numpy as jnp
from config import no_of_qubits, no_of_classes
import jax


def preprocess_data(dataset='cifar', encoding_mode='vanilla'):

    if dataset == 'cifar':
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Preprocess the dataset
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        # Resize images to match quantum circuit requirements
        resize_dim = int(2 ** (no_of_qubits / 2))  # 32x32 for 10 qubits
        x_train = tf.image.resize(x_train, (resize_dim, resize_dim)).numpy()
        x_test = tf.image.resize(x_test, (resize_dim, resize_dim)).numpy()

        # Convert to grayscale
        x_train = np.mean(x_train, axis=-1)  # Shape: (50000, 32, 32)
        x_test = np.mean(x_test, axis=-1)  # Shape: (10000, 32, 32)

        # Flatten to 1D
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Normalize to unit vectors
        x_train = x_train / np.sqrt(np.sum(x_train ** 2, axis=-1, keepdims=True))
        x_test = x_test / np.sqrt(np.sum(x_test ** 2, axis=-1, keepdims=True))

        # One-hot encode test labels
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=no_of_classes)

    elif dataset == 'mnist':
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize pixel values
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # Set mean based on encoding mode
        if encoding_mode == 'vanilla':
            mean = 0
        elif encoding_mode == 'mean':
            mean = jnp.mean(x_train, axis=0)
        elif encoding_mode == 'half':
            mean = 0.5

        # Normalize dataset
        x_train = x_train - mean
        x_test = x_test - mean

        # Resize images to match quantum circuit
        x_train = tf.image.resize(
            x_train[..., tf.newaxis],
            (int(2 ** (no_of_qubits / 2)), int(2 ** (no_of_qubits / 2)))
        ).numpy()[..., 0].reshape(-1, 2 ** no_of_qubits)
        x_test = tf.image.resize(
            x_test[..., tf.newaxis],
            (int(2 ** (no_of_qubits / 2)), int(2 ** (no_of_qubits / 2)))
        ).numpy()[..., 0].reshape(-1, 2 ** no_of_qubits)

        # Normalize to unit vectors
        x_train = x_train / jnp.sqrt(jnp.sum(x_train ** 2, axis=-1, keepdims=True))
        x_test = x_test / jnp.sqrt(jnp.sum(x_test ** 2, axis=-1, keepdims=True))

        # One-hot encode test labels
        y_test = jax.nn.one_hot(y_test, no_of_classes)

    else:
        raise ValueError("Dataset must be 'cifar' or 'mnist'")

    print(f"{dataset.upper()} - Training data shape:", x_train.shape)
    print(f"{dataset.upper()} - Testing data shape:", x_test.shape)
    print(f"{dataset.upper()} - Training labels shape:", y_train.shape)
    print(f"{dataset.upper()} - Testing labels shape:", y_test.shape)

    return x_train, y_train, x_test, y_test