from typing import Callable

import numpy as np
from numba import jit


# TODO: Implement vectorized version
def _compare_in_dims(value: np.array, compared: np.array) -> np.array:
    ...


def collect_non_dominated_solutions(solutions: np.array, objective: Callable[[np.array], np.array]) -> np.array:
    """

    :param solutions:
    :param objective:
    :return:
    """
    values = np.apply_along_axis(objective, 0, solutions)
    non_dominated_solutions = []

    for solution in solutions:
        non_dominated_solutions.append(
            np.all(
                np.logical_or(
                    # TODO: Expand to multiple dimensions
                    np.less_equal(solution[0], solutions[:, 0]),
                    np.less_equal(solution[1], solutions[:, 1])
                )
            )
        )

    return solutions[np.where(non_dominated_solutions == True)]
