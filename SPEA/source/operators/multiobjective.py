from typing import Callable

import numpy as np
from numba import jit

_OPTIMIZATION_MODE_SELECTION_MAPPING = {
    "min": np.less_equal,
    "max": np.greater_equal,
    "strict_min": np.less,
    "strict_max": np.greater,
}


@jit
def _compare_in_dims(
    single_solution: np.array, compared_solutions: np.array, comparison_operation: Callable[[np.array, np.array], bool],
) -> np.array:
    """
    Compares array representing single solutions to population of solutions with any callable comparison function

    example:
        >>> results = _compare_in_dims(np.array([5, 5]), np.array([[1, 1], [10, 10], [1, 10], [10, 1]]), np.less_equal)

    results will be equal to:
            `[[False, False],
              [True, True],
              [False, True],
              [True, False]]`

    :param single_solution: array in shape (N, ) where N is number of dimensions in optimization objective
    :param compared_solutions: population array in shape (POPULATION_SIZE, N, ) to be compared
    :param comparison_operation: callable used to compare, must handle arrays and return truth values

    :return: array of truth values in shape (POPULATION_SIZE, N, )
             where each element of array corresponds to comparison in solution n dim item to population
             member with it's index and corresponding n dim
    """
    results = [
        comparison_operation(single_solution, compared_solutions[item_index, :])
        for item_index in range(compared_solutions.shape[0])
    ]

    return np.array(results)


def is_non_dominated_solution(single_solution: np.array, compared_solutions: np.array, mode: str) -> bool:
    """
    Boolean function testing if solution in dominated in Pareto's sense
    Returns true if there no solution's in population with better objective value in each dimension

    :param single_solution: solution to compare
    :param compared_solutions: array of population solution will be compared to
    :param mode: optimization mode, valid options are `min` or `max`

    :return: True is solutions in non dominated in Pareto's sense
    """
    return np.all(
        np.any(
            _compare_in_dims(single_solution, compared_solutions, _OPTIMIZATION_MODE_SELECTION_MAPPING[mode],), axis=1,
        )
    )


def is_dominated_solution(single_solution: np.array, compared_solutions: np.array, mode: str) -> bool:
    """
    :return: Logical inverse of is_non_dominated_solution
    """
    return not is_non_dominated_solution(single_solution, compared_solutions, mode)


def assign_pareto_strength(single_solution: np.array, compared_solutions: np.array, mode: str) -> int:
    """
    Assigns strength to each solution based on the number
    of other solutions it dominates over in it's population,
    lowest value is 1, because solution always dominates itself

    :param single_solution: solution to compare
    :param compared_solutions: array of population solution will be compared to
    :param mode: optimization mode, valid options are `min` or `max`

    :return: strength corresponding to given single_solution in given population
    """
    return np.count_nonzero(
        np.all(
            _compare_in_dims(single_solution, compared_solutions, _OPTIMIZATION_MODE_SELECTION_MAPPING[mode]),
            axis=1,
        )
    )


def strength_binary_tournament_selection(solutions: np.array, mating_pool_size: int, mode: str) -> np.array:
    """
    Creates mating pool using binary tournament selection, where fitness is based on Pareto strength
    Two randomly drawn solutions are selected and compared to each other,
    fitter solution of each pair gets selected into mating pool

    :param solutions: array of N solutions, each is solution is array
    :param mating_pool_size: size of output mating pool
    :param mode: optimization mode

    :return: indices of selected solutions
    """
    candidates = np.random.randint(0, solutions.shape[0], mating_pool_size*2)
    strengths = np.apply_along_axis(assign_pareto_strength, 1, np.take(solutions, candidates, axis=0), solutions, mode)
    selected = np.argmax(strengths.reshape(mating_pool_size, 2), axis=1)
    selected += np.arange(0, mating_pool_size * 2, 2)
    return np.take(candidates, selected)


def strength_n_fittest_selection(solutions: np.array, mating_pool_size: int, mode: str) -> np.array:
    """
    Creates mating pool using n fittest selection, where fitness is based on Pareto strength
    Selects N fittest solutions from entire population, where N is chosen mating pool size

    :param solutions: array of N solutions, each solution is an array
    :param mating_pool_size: size of output mating pool
    :param mode: optimization mode

    :return: indices of selected solutions
    """
    strengths = np.apply_along_axis(assign_pareto_strength, 1, solutions, solutions, mode)
    return np.argsort(-strengths, axis=0)[:mating_pool_size]
