import numpy as np
from ...core import Tensor

class DataLoader:
    def __init__(self, data, target, batch_size=1, shuffle=False, collate_fn=None):
        assert len(data) == len(target), "Data and target must have the same number of samples"
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or self.default_collate
        self.num_samples = len(data)
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.index = 0
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.index >= self.num_samples:
            raise StopIteration
        batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch_data = [self.data[i] for i in batch_indices]
        batch_target = [self.target[i] for i in batch_indices]
        self.index += self.batch_size
        return self.collate_fn(batch_data, batch_target)

    def __len__(self):
        return self.num_batches

    @staticmethod
    def default_collate(batch_data, batch_target):
        inputs = Tensor(np.array(batch_data))
        targets = Tensor(np.array(batch_target))
        return inputs, targets
