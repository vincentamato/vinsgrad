import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple

class VisionDataset(ABC):
    """
    Abstract base class for vision datasets.
    """

    def __init__(self, root: str, transform: Callable = None):
        """
        Initialize the VisionDataset.

        Args:
            root (str): Root directory of the dataset.
            transform (Callable, optional): A function/transform to apply on the data.
        """
        self.root = root
        self.transform = transform

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: (sample, target) where target is the index of the target class.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        pass

    def _check_exists(self) -> bool:
        """
        Check if the dataset exists in the root directory.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        return os.path.exists(self.root)

    @abstractmethod
    def download(self) -> None:
        """
        Download the dataset if it doesn't exist in the root directory.
        """
        pass

    def apply_transform(self, sample: Any) -> Any:
        """
        Apply the transform to the sample.

        Args:
            sample (Any): The sample to transform.

        Returns:
            Any: The transformed sample.
        """
        if self.transform:
            return self.transform(sample)
        return sample