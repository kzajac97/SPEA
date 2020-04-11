from typing import Callable, Tuple

import numpy as np
from tqdm import tqdm

from source.pareto_set import ParetoSet
from source.operators.continuous import vectorized_crossover, vectorized_mutation
from source.operators.multiobjective import is_non_dominated_solution, strength_n_fittest_selection, strength_binary_tournament_selection
from source.utils.distance import range_size


class SPEAOptimizer:
    """
    """

    # TODO: Docstring
    def __init__(self, objective: Callable[[np.array], np.array], n_dim: int, mode: str):
        """

        :param objective:
        :param n_dim:
        :param mode:
        """
        self._objective = objective
        self._optimization_mode = mode
        self._n_dim = n_dim
        self._external_set = None
        self._selection_operator = {
            "n_fittest": strength_n_fittest_selection,
            "binary_tournament": strength_binary_tournament_selection
        }

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

    # TODO: Docstring
    def _collect_all_non_dominated_individuals(self, population: np.array) -> np.array:
        """

        :param population:
        :return:
        """
        solutions = np.apply_along_axis(self._objective, 1, population)
        is_non_dominated = np.array(
            [is_non_dominated_solution(solution, solutions, mode=self._optimization_mode,) for solution in solutions]
        )

        return population[np.where(is_non_dominated == True)]

    @staticmethod
    def mutate_population(population: np.array, mutation_rate: float, mutation_strength: float):
        """
        Perform mutation on randomly chosen subset of current population

        :param population: array of current solutions
        :param mutation_rate: fraction of solutions mutation will be applied to, must be in range <0, 1>
        :param mutation_strength: factor of mutation strength, should be proportional to search space size

        :return: array of mutated solutions
        """
        population = np.copy(population)
        to_mutate = np.random.randint(0, len(population), int(mutation_rate * len(population)))
        population[to_mutate] = vectorized_mutation(population[to_mutate], mutation_strength)

        return population

    def create_offspring(
            self, population: np.array, mating_pool_size: int, n_offspring: int, selection_operator: str
    ) -> np.array:
        """
        Create offspring solutions using crossover

        :param population: array of current solutions
        :param mating_pool_size: number of solutions taking part in crossover
        :param n_offspring: number of solutions created by crossover
        :param selection_operator: selection algorithm

        :return: array of generated offspring solutions
        """
        solutions = np.apply_along_axis(self._objective, 1, population)
        mating_pool = self._selection_operator[selection_operator](solutions, mating_pool_size, mode=self._optimization_mode)
        return vectorized_crossover(population[mating_pool, :], n_offspring)

    # TODO: Move parameters to __init__, create better interface, docstring
    def optimize(
        self,
        num_epochs: int,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        selection_operator: str,
        reducing_period: int,
        search_range: Tuple[Tuple[float, float], ...],
        mutation_strength: float = 0.1,
        relative_mutation_strength: bool = True,
        clustering_parameters: dict = None,
        silent: bool = True,
        logging: bool = False,  # TODO: Add logging options
    ):
        population = self._init_population(population_size, initial_search_range=search_range)
        self._external_set = ParetoSet(reducing_period=reducing_period, model_kwargs=clustering_parameters)
        mutation_strength = (
            range_size(search_range) * mutation_strength if relative_mutation_strength else mutation_strength
        )

        for epoch in tqdm(range(num_epochs), disable=silent):
            pareto_solutions = self._collect_all_non_dominated_individuals(population)
            # update pareto set
            self._external_set.update(pareto_solutions)
            # use pareto set
            population = np.concatenate([population, self._external_set.callback(epoch)], axis=0)
            # crossover
            population = self.create_offspring(population, int(crossover_rate * len(population)), population_size, selection_operator)
            # mutation
            population = self.mutate_population(population, mutation_rate, mutation_strength)
            #

        return population

    def _check_args(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        ...
