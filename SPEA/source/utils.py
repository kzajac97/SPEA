import string
from typing import Tuple

import numpy as np
from numba import jit

X_ALPHABET_SHIFT = 23


@jit
def row_wise_multiply(x: np.array, y: np.array) -> np.array:
    """
    Perform row wise multiplication

    Example:
       >>> row_wise_multiply([1, -1], [[-10, 10], [-10, 10]])
       ... [[-10, 10], [10, -10]]
    """
    return np.apply_along_axis(np.multiply, 0, x, y)


def mean_search_space_size(search_range: Tuple[Tuple[float, float], ...]):
    """

    :param search_range:
    :return:
    """
    search_range = np.array(search_range)
    return np.mean(search_range[:, 1] - search_range[:, 0])


def dims_to_column_names(array: np.array) -> list:
    """
    :return: conventional axes names for ND array
    """
    return list(string.ascii_lowercase[X_ALPHABET_SHIFT: X_ALPHABET_SHIFT + array.ndim])