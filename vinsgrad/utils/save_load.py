import pickle
import gzip
import os
from datetime import datetime
from typing import Any, Optional

def save(obj: Any, model_name: str, epoch: Optional[int] = None, is_best: bool = False, dir_path: str = 'checkpoints') -> str:
    """
    Save an object to a file in the specified directory.
    The object can be a state_dict or a dictionary containing multiple state_dicts.
    Creates the directory if it doesn't exist.
    
    Args:
        obj (Any): The object to save.
        model_name (str): Name of the model (e.g., 'mnist_mlp').
        epoch (Optional[int]): Current epoch number (optional).
        is_best (bool): Whether this is the best model so far (optional).
        dir_path (str): Directory path where the object should be saved.
    
    Returns:
        str: The filename the object was saved to.
    """
    os.makedirs(dir_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_best:
        filename = f"{dir_path}/{model_name}_best.pkl.gz"
    elif epoch is not None:
        filename = f"{dir_path}/{model_name}_epoch{epoch}_{timestamp}.pkl.gz"
    else:
        filename = f"{dir_path}/{model_name}_{timestamp}.pkl.gz"

    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"Model saved to {filename}")
    
    return filename

def load(filename: str) -> Any:
    """
    Load an object from a file.
    Provides an informative error message if the file doesn't exist.
    
    Args:
        filename (str): The path to the file.
    
    Returns:
        Any: The loaded object.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No file found at {filename}. Please make sure the file exists and the path is correct.")
    
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)
