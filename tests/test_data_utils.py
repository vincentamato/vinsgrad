import unittest
import numpy as np
from vinsgrad.utils.data import DataLoader, random_split, SubsetDataset

class MockDataset:
    """
    A simple mock dataset that returns data and labels as tuples.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

class TestDataUtils(unittest.TestCase):
    
    def setUp(self):
        """
        Set up the test environment.
        """
        self.data = np.arange(100).reshape(50, 2)
        self.labels = np.arange(50)
        self.dataset = MockDataset(self.data, self.labels)

    def test_dataloader(self):
        """
        Test the DataLoader class.
        """
        dataloader = DataLoader(self.dataset, batch_size=10, shuffle=True)
        batches = list(dataloader)
        
        self.assertEqual(len(batches), 5)
        for data, labels in batches:
            self.assertEqual(data.shape[0], 10)
            self.assertEqual(labels.shape[0], 10)
    
    def test_random_split(self):
        """
        Test the random_split function.
        """
        lengths = [30, 20]
        subsets = random_split(self.dataset, lengths)
        
        self.assertEqual(len(subsets), 2)
        self.assertEqual(len(subsets[0]), 30)
        self.assertEqual(len(subsets[1]), 20)
    
    def test_subset_dataset(self):
        """
        Test the SubsetDataset class.
        """
        indices = np.random.permutation(len(self.dataset))[:10]
        subset = SubsetDataset(self.dataset, indices)
        
        self.assertEqual(len(subset), 10)
        for i in range(10):
            data, label = subset[i]
            self.assertTrue(np.array_equal(data, self.data[indices[i]]))
            self.assertEqual(label, self.labels[indices[i]])

if __name__ == '__main__':
    unittest.main()
