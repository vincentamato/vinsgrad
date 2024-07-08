import numpy as np
from ..core import Tensor

class Compose:
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        """
        Initializes the Compose transform.
        
        Args:
            transforms (list of callable): List of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, img):
        """
        Applies the composed transforms to the image.
        
        Args:
            img (numpy.ndarray): The input image.
        
        Returns:
            numpy.ndarray: The transformed image.
        """
        for t in self.transforms:
            img = t(img)
        return img

class ToTensor:
    """
    Converts a numpy.ndarray to a Tensor.
    """

    def __call__(self, image):
        """
        Converts the input image to a Tensor.
        
        Args:
            image (numpy.ndarray): The input image.
        
        Returns:
            Tensor: The image converted to a Tensor.
        
        Raises:
            TypeError: If the input is not a numpy ndarray.
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, None]
            elif image.ndim == 3 and image.shape[2] == 1:
                image = image[:, :, 0]
                image = image[:, :, None]
            image = image.transpose((2, 0, 1))
            image = image.astype(np.float32) / 255.0
            return Tensor(image)
        else:
            raise TypeError('Input image should be a numpy ndarray. Got {}'.format(type(image)))
        
class Normalize:
    """
    Normalizes a Tensor with mean and standard deviation.
    """
    def __init__(self, mean, std):
        """
        Initializes the Normalize transform.
        
        Args:
            mean (list or tuple): The mean for each channel.
            std (list or tuple): The standard deviation for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image):
        """
        Normalizes the input Tensor.
        
        Args:
            image (Tensor): The input image tensor.
        
        Returns:
            Tensor: The normalized image tensor.
        
        Raises:
            TypeError: If the input is not a Tensor.
        """
        if isinstance(image, Tensor):
            mean = np.array(self.mean).reshape(-1, 1, 1)
            std = np.array(self.std).reshape(-1, 1, 1)
            return Tensor((image.data - mean) / std)
        else:
            raise TypeError('Input image should be a Tensor. Got {}'.format(type(image)))
        
class Flatten:
    """
    Flattens a Tensor.
    """
    def __call__(self, img):
        """
        Flattens the input image tensor.
        
        Args:
            img (Tensor): The input image tensor.
        
        Returns:
            Tensor: The flattened image tensor.
        """
        return img.reshape(-1)
