import logging
from typing import Callable, Optional

import numpy as np
from numba import jit

logger = logging.getLogger(__name__)

_OPTIMIZATION_MODE_SELECTION_MAPPING = {
    "min": np.less_equal,
    "max": np.greater_equal,
}


@jit
def _compare_in_dims(
    single_solution: np.array,
    compared_solutions: np.array,
    comparison_operation: Callable[[np.array, np.array], bool]
) -> np.array:
    """
    """
    results = [comparison_operation(single_solution,
                                    compared_solutions[item_index, :])
               for item_index in range(compared_solutions.shape[0])]

    return np.array(results)


@jit
def is_non_dominated_solution(single_solution: np.array, compared_solutions: np.array, mode: str) -> Optional[bool]:
    """
    Boolean function testing if solution in dominated in Pareto's sense

    :param single_solution: solution to compare
    :param compared_solutions: array of population solution will be compared to
    :param mode: optimization mode, valid options are `min` or `max`

    :return: True is solutions in non dominated in Pareto's sense
    """
    if mode not in _OPTIMIZATION_MODE_SELECTION_MAPPING.keys():
        logger.error(f"{mode} is not valid optimization mode!")
        return

    return np.all(np.any(
            _compare_in_dims(
                single_solution,
                compared_solutions,
                _OPTIMIZATION_MODE_SELECTION_MAPPING[mode]
            ), axis=1))


@jit
def is_dominated_solution(single_solution: np.array, compared_solutions: np.array, mode: str) -> Optional[bool]:
    """
    :return: Logical inverse of is_non_dominated_solution
    """
    if mode not in _OPTIMIZATION_MODE_SELECTION_MAPPING.keys():
        logger.error(f"{mode} is not valid optimization mode!")
        return

    return not is_non_dominated_solution(single_solution, compared_solutions, mode)


def collect_non_dominated_solutions(single_solution: np.array, compared_solutions: np.array, mode: str) -> np.array:
    """

    :param single_solution:
    :param compared_solutions:
    :param mode:
    :return:
    """
    is_dominated = np.any(_compare_in_dims(
        single_solution,
        compared_solutions,
        comparison_operation=_OPTIMIZATION_MODE_SELECTION_MAPPING[mode]
    ), axis=1)

    return compared_solutions[np.where(np.logical_not(is_dominated))]


def collect_dominated_solutions(single_solution: np.array, compared_solutions: np.array, mode: str) -> np.array:
    is_dominated = np.any(_compare_in_dims(
        single_solution,
        compared_solutions,
        comparison_operation=_OPTIMIZATION_MODE_SELECTION_MAPPING[mode]
    ), axis=1)

    return compared_solutions[np.where(is_dominated)]


def assign_pareto_strength(single_solution: np.array, compared_solutions: np.array, mode: str) -> int:
    dominated_solutions = collect_dominated_solutions(
        single_solution,
        compared_solutions,
        mode=mode,
    )

    return dominated_solutions.shape[0]
