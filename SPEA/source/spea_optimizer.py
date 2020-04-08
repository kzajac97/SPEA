from typing import Callable, Tuple

import numpy as np

from source.operators.continuous import crossover, mutation
from source.operators.multiobjective import is_non_dominated_solution, collect_non_dominated_solutions, assign_pareto_strength


class SPEAOptimizer:
    """
    """

    def __init__(self, objective: Callable[[np.array], np.array], n_dim: int, mode: str):
        """

        :param objective:
        :param n_dim:
        :param mode:
        """
        self._objective = objective
        self._optimization_mode = mode
        self._n_dim = n_dim

    def _init_population(self, population_size: int, initial_search_range: Tuple[Tuple[float, float], ...]) -> np.array:
        """
        Initializes population of solutions:

        Example:
            >>> self._init_population(1000, ((0, 10), (10, 20), (-10, 0)))

        Initializes population for 3D objective, with ranges [0, 10], [10, 20] and [-10, 0] in each dimension

        :param population_size: number of individuals to generate
        :param initial_search_range: tuple of ranges in all dimensions

        :return: initialized vector of individuals. with shape (POPULATION_SIZE, N_DIM)
        """
        population = np.random.random_sample(population_size * self._n_dim).reshape(population_size, self._n_dim)

        for dimension_index, (lower_bound, upper_bound) in enumerate(initial_search_range):
            population[:, dimension_index] = (upper_bound - lower_bound) * population[:, dimension_index] + lower_bound

        return population

    def _collect_all_non_dominated_individuals(self, population: np.array) -> np.array:
        """

        :param population:
        :return:
        """
        solutions = np.apply_along_axis(self._objective, 1, population)
        is_non_dominated = np.array([is_non_dominated_solution(
            solution,
            solutions,
            mode=self._optimization_mode,
        ) for solution in solutions])

        return population[np.where(is_non_dominated == True)]

    def optimize(self, steps: int, initial_population_size: int):
        population = self._init_population(initial_population_size, ((0, 10), ))

        for step in range(steps):
            ...
