import os
import gzip
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_mnist_images(filename):
    """Load MNIST images from a gzip file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} не знайдено. Завантажте файл із офіційного джерела.")
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    """Load MNIST labels from a gzip file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} не знайдено. Завантажте файл із офіційного джерела.")
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

def prepare_data(dataset_path, batch_size=64):
    """Load, preprocess data and return DataLoader objects."""
    train_images = load_mnist_images(os.path.join(dataset_path, 'train-images-idx3-ubyte.gz'))
    train_labels = load_mnist_labels(os.path.join(dataset_path, 'train-labels-idx1-ubyte.gz'))
    test_images = load_mnist_images(os.path.join(dataset_path, 't10k-images-idx3-ubyte.gz'))
    test_labels = load_mnist_labels(os.path.join(dataset_path, 't10k-labels-idx1-ubyte.gz'))

    # Convert to tensors
    train_images_tensor = torch.tensor(train_images)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_images_tensor = torch.tensor(test_images)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
