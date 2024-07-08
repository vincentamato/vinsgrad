from typing import List, Any

class SubsetDataset:
    """
    A dataset that is a subset of another dataset.

    Args:
        dataset (Any): The original dataset.
        indices (List[int]): Indices to select from the original dataset.
    """

    def __init__(self, dataset: Any, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int) -> Any:
        """
        Get an item from the subset.

        Args:
            idx (int): Index of the item to get.

        Returns:
            Any: The item at the specified index.
        """
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        """
        Get the length of the subset.

        Returns:
            int: The number of items in the subset.
        """
        return len(self.indices)