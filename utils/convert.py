"""The convert labels implementation.
"""
import numpy as np


def cvt2cat(array, num_class=None, dtype='float32'):
    """The operation for convert to categorical labels.

    Args:
        array (np.ndarray): The input labels.
        num_class (int, optional): The number of class. Defaults to None.
        dtype (str): The data type.

    Returns:
        np.ndarray: The categorical labels.
    """
    assert array.ndim == 1, "Array should have 1-dimension."

    if num_class is None:
        num_class = array.max()+1

    return np.eye(num_class, dtype=dtype)[array]


def cvt2sps(array):
    """The operation for convert to sparse labels.

    Args:
        array (np.ndarray): The input labels.

    Returns:
        np.ndarray: The sparse labels.
    """
    assert array.ndim == 2, "Array should have 2-dimension."

    return np.argmax(array, axis=1)
