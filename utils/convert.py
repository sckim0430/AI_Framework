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

    if not num_class:
        num_class = np.max(array)+1

    batch_size = array.shape[0]

    categorical = np.zeros((batch_size, num_class), dtype=dtype)

    for idx, a in enumerate(array):
        categorical[idx,a] = 1

    return categorical


def cvt2sps(array):
    """Convert to sparse label.

    Args:
        array (np.ndarray): The input labels.

    Returns:
        np.ndarray: The sparse labels.
    """
    assert array.ndim == 2, "Array should have 2-dimension."

    sparse = np.argmax(array, axis=1)

    return sparse
