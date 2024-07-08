import numpy as np
from typing import List, Any, Tuple
from .subset_dataset import SubsetDataset


def random_split(dataset: Any, lengths: List[int]) -> Tuple[SubsetDataset, ...]:
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Args:
        dataset (Any): Dataset to be split.
        lengths (List[int]): Lengths of the splits.

    Returns:
        Tuple[SubsetDataset, ...]: A tuple of SubsetDatasets.

    Raises:
        ValueError: If sum of lengths does not equal the length of the dataset.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of lengths does not equal the length of the dataset")

    indices = np.random.permutation(len(dataset))
    return tuple(
        SubsetDataset(dataset, indices[offset - length : offset])
        for offset, length in zip(np.cumsum(lengths), lengths)
    )