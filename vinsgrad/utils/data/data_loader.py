import numpy as np
from typing import Callable, Any, List, Tuple
from ...core import Tensor

class DataLoader:
    """
    DataLoader class for loading data in batches.
    """
    
    def __init__(self, 
                 dataset: Any,
                 batch_size: int = 1, 
                 shuffle: bool = False, 
                 collate_fn: Callable = None) -> None:
        """
        Initializes the DataLoader.
        
        Args:
            dataset (Any): The dataset to load from. Should implement __getitem__ and __len__.
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the data before each epoch.
            collate_fn (Callable): Function to collate a batch of data.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or self.default_collate
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Initializes the iterator for the DataLoader.
        
        Returns:
            DataLoader: The DataLoader instance.
        """
        self.index = 0
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """
        Retrieves the next batch of data.
        
        Returns:
            Tuple[Tensor, Tensor]: A batch of input data and target labels.
        
        Raises:
            StopIteration: If there are no more batches to return.
        """
        if self.index >= self.num_samples:
            raise StopIteration
        batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.index += self.batch_size
        return self.collate_fn(batch)

    def __len__(self):
        """
        Returns the number of batches.
        
        Returns:
            int: The number of batches.
        """
        return self.num_batches

    @staticmethod
    def default_collate(batch: List[Tuple[Any, Any]]) -> Tuple[Tensor, Tensor]:
        """
        Default collate function to combine a list of samples into a batch.
        
        Args:
            batch (List[Tuple[Any, Any]]): A list of tuples (data, target).
        
        Returns:
            Tuple[Tensor, Tensor]: A batch of input data and target labels.
        """
        data = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        if isinstance(data[0], Tensor):
            data = Tensor(np.stack([d.data for d in data]))
        else:
            data = Tensor(np.array(data))
        
        if isinstance(targets[0], Tensor):
            targets = Tensor(np.stack([t.data for t in targets]))
        else:
            targets = Tensor(np.array(targets))
        
        return data, targets
