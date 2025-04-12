from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoHumanOrWorm
from matplotlib import pyplot as plt
import seaborn as sns
from qiskit_algorithms.utils import algorithm_globals

def load_data(data_used, data_size):
    subset_size = 150
    subset_size1 = 50
    algorithm_globals.random_seed = 123

    if data_used == "iris":
        iris_data = load_iris()
        features = iris_data.data
        labels = iris_data.target
        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette="tab10")
        plt.title("IRIS Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    elif data_used == "synthetic":
        num_samples = 1000
        num_features = 4
        features = 2 * algorithm_globals.random.random([num_samples, num_features]) - 1
        labels = 1 * (np.sum(features, axis=1) >= 0)
        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette="tab10")
        plt.title("Synthetic Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    elif data_used == "mnist_keras":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if data_size == "small":
            x_train = x_train[:subset_size]
            y_train = y_train[:subset_size]
            x_test = x_test[:subset_size1]
            y_test = y_test[:subset_size1]
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        scaler = StandardScaler()
        x_train_flattened = scaler.fit_transform(x_train_flattened)
        x_test_flattened = scaler.transform(x_test_flattened)
        pca = PCA(n_components=4)
        x_train_pca = pca.fit_transform(x_train_flattened)
        x_test_pca = pca.transform(x_test_flattened)
        features = x_train_pca
        labels = y_train
        plt.rcParams["figure.figsize"] = (8, 8)
        sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train, palette="tab10", legend="full", s=60, alpha=0.7)
        plt.title("MNIST Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    elif data_used == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        if data_size == "small":
            x_train = x_train[:subset_size]
            y_train = y_train[:subset_size]
            x_test = x_test[:subset_size1]
            y_test = y_test[:subset_size1]
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        scaler = StandardScaler()
        x_train_flattened = scaler.fit_transform(x_train_flattened)
        x_test_flattened = scaler.transform(x_test_flattened)
        pca = PCA(n_components=4)
        x_train_pca = pca.fit_transform(x_train_flattened)
        x_test_pca = pca.transform(x_test_flattened)
        features = x_train_pca
        labels = y_train
        plt.rcParams["figure.figsize"] = (8, 8)
        sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train, palette="tab10", legend="full", s=60, alpha=0.7)
        plt.title("Fashion MNIST Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    elif data_used == "cifar":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        if data_size == "small":
            x_train = x_train[:subset_size]
            y_train = y_train[:subset_size]
            x_test = x_test[:subset_size1]
            y_test = y_test[:subset_size1]
        x_train_flattened = x_train.reshape(x_train.shape[0], -1)
        x_test_flattened = x_test.reshape(x_test.shape[0], -1)
        scaler = StandardScaler()
        x_train_flattened = scaler.fit_transform(x_train_flattened)
        x_test_flattened = scaler.transform(x_test_flattened)
        pca = PCA(n_components=4)
        x_train_pca = pca.fit_transform(x_train_flattened)
        x_test_pca = pca.transform(x_test_flattened)
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        features = x_train_pca
        labels = y_train
        plt.rcParams["figure.figsize"] = (8, 8)
        sns.scatterplot(x=x_train_pca[:, 0], y=x_train_pca[:, 1], hue=y_train, palette="tab10", legend="full", s=60, alpha=0.7)
        plt.title("CIFAR-10 Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    elif data_used == "mnist":
        mnist_data = load_digits()
        features = mnist_data.data
        labels = mnist_data.target
        features_pca = PCA(n_components=4).fit_transform(features)
        features = features_pca
        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=labels, palette="tab10")
        plt.title("MNIST Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

    elif data_used == "genomics":
        train_dataset = DemoHumanOrWorm(split='train', version=0)
        train_data_list = list(train_dataset)
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        encoded_sequences = []
        labels = []
        for sequence, label in train_data_list:
            encoded_sequence = []
            for nucleotide in sequence:
                encoded_nucleotide = [0] * 4
                if nucleotide in nucleotide_map:
                    index = nucleotide_map[nucleotide]
                    encoded_nucleotide[index] = 1
                encoded_sequence.append(encoded_nucleotide)
            encoded_sequences.append(encoded_sequence)
            labels.append(label)
        features_3D = np.array(encoded_sequences)
        labels = np.array(labels)
        features_reshaped = features_3D.reshape(features_3D.shape[0], -1)
        features = PCA(n_components=4).fit_transform(features_reshaped)
        plt.rcParams["figure.figsize"] = (6, 6)
        sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=labels, palette="tab10")
        plt.title("Encoded Sequences")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    else:
        raise ValueError("No Data Set Selected!")

    print(f"Dataset used is: {data_used}")
    alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = train_test_split(
        features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
    )
    return alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels