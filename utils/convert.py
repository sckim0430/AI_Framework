"""Implementation of convert labels.
"""
import numpy as np

def cvt2cat(array):
    """Convert to categori label.

    Args:
        array (np.ndarray): The input labels.

    Returns:
        np.ndarray : The categori labels.
    """
    assert array.ndim == 2, "Array should have 2-dimension."
    return np.argmax(array,axis=1)