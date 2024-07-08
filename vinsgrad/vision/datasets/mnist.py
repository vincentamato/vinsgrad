import numpy as np
import os
from urllib.request import urlretrieve
from ...core import Tensor
from ..vision_dataset import VisionDataset

class MNIST(VisionDataset):
    """
    MNIST dataset class for loading MNIST data.
    """

    def __init__(self, root='./mnist_data', train=True, download=True, transform=None):
        """
        Initializes the MNIST dataset.
        
        Args:
            root (str): The root directory for the dataset.
            train (bool): Whether to load the training set. If False, loads the test set.
            download (bool): Whether to download the dataset if it does not exist.
            transform (callable, optional): A function/transform to apply to the data.
        """
        super().__init__(root, transform)
        self.train = train
        self.filename = 'mnist.npz'
        self.filepath = os.path.join(self.root, self.filename)
        self.url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

        self.data = None
        self.targets = None

        if download:
            self.download()

        self.load_data()

    def download(self):
        """
        Downloads the MNIST dataset if it does not already exist.
        """
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)
        print('Downloading MNIST data...')
        try:
            urlretrieve(self.url, self.filepath)
            print(f'Downloaded MNIST data to {self.filepath}')
        except Exception as e:
            print(f'Failed to download MNIST data from {self.url}')
            raise Exception('Please check your internet connection and try again.') from e

    def load_data(self):
        """
        Loads the MNIST data from the .npz file.
        """
        with np.load(self.filepath, allow_pickle=True) as f:
            if self.train:
                self.data = f['x_train']
                self.targets = f['y_train']
            else:
                self.data = f['x_test']
                self.targets = f['y_test']

    def __getitem__(self, index):
        """
        Retrieves the data and target at the specified index.
        
        Args:
            index (int): The index of the data to retrieve.
        
        Returns:
            Tuple[Tensor, Tensor]: The data and the one-hot encoded target.
        """
        img, target = self.data[index], int(self.targets[index])
        
        if self.transform:
            img = self.transform(img)
        
        target_one_hot = np.zeros(10, dtype=np.float32)
        target_one_hot[target] = 1.0
        target_tensor = Tensor(target_one_hot)
        
        return img, target_tensor

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)
