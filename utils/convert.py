"""Implementation of convert labels.
"""
import numpy as np


def cvt2cat(array, num_class=None, dtype='float32'):
    """Convert to categorical label.

    Args:
        array (np.ndarray): The input labels.
        num_class (int, optional): Number of the class. Defaults to None.
        dtype (str): The data type.

    Returns:
        np.ndarray: The categorical labels.
    """
    assert array.ndim == 1, "Array should have 1-dimension."

    if not num_class:
        num_class = array.max()+1

    return np.eye(num_class, dtype=dtype)[array]


def cvt2sps(array):
    """Convert to sparse label.

    Args:
        array (np.ndarray): The input labels.

    Returns:
        np.ndarray: The sparse labels.
    """
    assert array.ndim == 2, "Array should have 2-dimension."

    return np.argmax(array, axis=1)
