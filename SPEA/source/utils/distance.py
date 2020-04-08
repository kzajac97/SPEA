from typing import Tuple

import numpy as np


def range_size(search_range: Tuple[Tuple[float, float], ...]):
    """

    :param search_range:
    :return:
    """
    search_range = np.array(search_range)
    return np.prod(search_range[:, 1] - search_range[:, 0])
