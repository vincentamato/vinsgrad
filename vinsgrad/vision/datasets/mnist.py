import numpy as np
import os
from urllib.request import urlretrieve
from ...core import Tensor

class MNIST:
    def __init__(self, root='./data', download=True, transform=None):
        self.root = root
        self.filename = 'mnist.npz'
        self.filepath = os.path.join(self.root, self.filename)
        self.url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.transform = transform

        if download:
            self.download()

        self.load_data()

    def download(self):
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(self.filepath):
            print(f'Downloading MNIST data...')
            try:
                urlretrieve(self.url, self.filepath)
                print(f'Downloaded MNIST data to {self.filepath}')
            except:
                print(f'Failed to download MNIST data from {self.url}')
                raise Exception('please check your internet connection and try again.')

    def load_data(self):
        with np.load(self.filepath, allow_pickle=True) as f:
            self.train_images = f['x_train'].astype(np.float32)
            self.train_labels = self.one_hot_encode(f['y_train'])
            self.test_images = f['x_test'].astype(np.float32)
            self.test_labels = self.one_hot_encode(f['y_test'])

    def one_hot_encode(self, labels, num_classes=10):
        return np.eye(num_classes)[labels]

    def apply_transform(self, images):
        if self.transform:
            transformed = [self.transform(img) for img in images]
            return np.array(transformed)
        return images

    def get_train_data(self):
        transformed_images = self.apply_transform(self.train_images)
        return transformed_images, self.train_labels

    def get_test_data(self):
        transformed_images = self.apply_transform(self.test_images)
        return transformed_images, self.test_labels
