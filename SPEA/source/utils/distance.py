from typing import Tuple

import numpy as np
from numba import jit


def range_size(search_range: Tuple[Tuple[float, float], ...]):
    """

    :param search_range:
    :return:
    """
    search_range = np.array(search_range)
    return np.prod(search_range[:, 1] - search_range[:, 0])


@jit
def row_wise_multiply(x: np.array, y: np.array) -> np.array:
    """
    Perform row wise multiplication

    Example:
       >>> row_wise_multiply([1, -1], [[-10, 10], [-10, 10]])
       ... [[-10, 10], [10, -10]]
    """
    return np.apply_along_axis(np.multiply, 0, x, y)
