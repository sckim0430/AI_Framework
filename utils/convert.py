"""The convert labels implementation.
"""
import numpy as np


def cvt2cat(array, num_class=None, dtype='float32'):
    """The operation for convert to categorical labels.

    Args:
        array (np.ndarray): The input labels.
        num_class (int, optional): The number of class. Defaults to None.
        dtype (str): The data type.

    Raises:
        ValueError: The array should have 1-dimension.

    Returns:
        np.ndarray: The categorical labels.
    """
    if array.ndim!=1:
        raise ValueError('The array should have 1-dimension.')

    if num_class is None:
        num_class = array.max()+1

    return np.eye(num_class, dtype=dtype)[array]


def cvt2sps(array):
    """The operation for convert to sparse labels.

    Args:
        array (np.ndarray): The input labels.

    Raises:
        ValueError: The array should have 2-dimension.

    Returns:
        np.ndarray: The sparse labels.
    """
    if array.ndim!=2:
        raise ValueError('The array should have 2-dimension.')

    return np.argmax(array, axis=1)
