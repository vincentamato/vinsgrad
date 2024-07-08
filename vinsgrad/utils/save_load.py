import pickle
import gzip
import os
from datetime import datetime

def save(obj, model_name, epoch=None, is_best=False):
    """
    Save an object to a file in the 'checkpoints' folder.
    The object can be a state_dict or a dictionary containing multiple state_dicts.
    Creates a 'checkpoints' folder if it doesn't exist.
    
    :param obj: The object to save
    :param model_name: Name of the model (e.g., 'mnist_mlp')
    :param epoch: Current epoch number (optional)
    :param is_best: Whether this is the best model so far (optional)
    """
    checkpoints_dir = 'checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_best:
        filename = f"{checkpoints_dir}/{model_name}_best.pkl.gz"
    elif epoch is not None:
        filename = f"{checkpoints_dir}/{model_name}_epoch{epoch}_{timestamp}.pkl.gz"
    else:
        filename = f"{checkpoints_dir}/{model_name}_{timestamp}.pkl.gz"

    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"Model saved to {filename}")

def load(filename):
    """
    Load an object from a file.
    Provides an informative error message if the file doesn't exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No file found at {filename}. Please make sure the file exists and the path is correct.")
    
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

# These functions should be part of your vinsgrad package