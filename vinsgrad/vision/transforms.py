import numpy as np
from ..core import Tensor

class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

class ToTensor:

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # For grayscale images, add a channel dimension
                image = image[:, :, None]
            elif image.ndim == 3 and image.shape[2] == 1:
                # For single channel images, ensure correct shape
                image = image[:, :, 0]  # Remove the redundant channel dimension
                image = image[:, :, None]  # Add back the single channel dimension
            # Convert from HWC to CHW format
            image = image.transpose((2, 0, 1))
            # Normalize the values to [0, 1]
            image = image.astype(np.float32) / 255.0
            return Tensor(image)
        else:
            raise TypeError('Input image should be a numpy ndarray. Got {}'.format(type(image)))
        
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        if isinstance(image, Tensor):
            mean = np.array(self.mean).reshape(-1, 1, 1)
            std = np.array(self.std).reshape(-1, 1, 1)
            return Tensor((image.data - mean) / std)
        else:
            raise TypeError('Input image should be a Tensor. Got {}'.format(type(image)))
        
class Flatten:

    def __call__(self, image):
        if isinstance(image, Tensor):
            return Tensor(image.data.flatten())
        else:
            raise TypeError('Input image should be a Tensor. Got {}'.format(type(image)))

