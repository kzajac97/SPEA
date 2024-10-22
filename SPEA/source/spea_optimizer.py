from pathlib import Path
from typing import Any, Callable, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation, KMeans, MeanShift, SpectralClustering
from tqdm import tqdm

from source.pareto_set import ParetoSet
from source.operators.continuous import vectorized_hypersphere_crossover, vectorized_gaussian_mutation
from source.operators.multiobjective import (
    is_non_dominated_solution,
    strength_n_fittest_selection,
    strength_binary_tournament_selection,
)
from source.operators.schedules import const_schedule, increasing_linear_schedule, decreasing_linear_schedule
from source.utils import array_subset, dims_to_column_names, flatten

_SELECTION_OPERATOR_MAPPING = {
    "n_fittest": strength_n_fittest_selection,
    "binary_tournament": strength_binary_tournament_selection,
}
_MUTATION_OPERATOR_MAPPING = {
    "gaussian": vectorized_gaussian_mutation,
}
_CROSSOVER_OPERATOR_MAPPING = {
    "center": vectorized_hypersphere_crossover,
}
_CLUSTER_METHOD_MAPPING = {
    "affinity_propagation": AffinityPropagation,
    "kmeans": KMeans,
    "mean_shift": MeanShift,
    "spectral": SpectralClustering,
}


class SPEAOptimizer:
    """
    Class holds implementation of Strength Pareto Evolutionary Algorithm.
    For more details on the algorithm see: README.md

    :param objective: multi objective function to optimize,
                      takes and returns numpy arrays where each input element is variable
                      and each output is one of component objectives
    :param n_dim: number of input dimensions of objective, it can not be deduced because of dynamic typing
    :param mode: optimization mode, valid options are `min`, `max`, `strict_min` or `strict_max`
    :param selection_operator: genetic operator preforming selection operation, str or callable
                               if str valid options are `n_fittest` and `binary_tournament`
                               if callable should, take in population to select from,
                               mating pool size and optimization mode
    :param mutation_operator: genetic operator preforming mutation, str or callable
                              if str valid options are `gaussian`
                              if callable should, take in population to mutate
                              (most likely a fraction of whole population)
                              and strength of mutation, for floating point encoded genes
    :param crossover_operator: genetic operator preforming crossover, str or callable
                               if str valid options are `center`
                               if callable should, take in mating pool to crossover
                               as paris of individuals to cross over
                               (most likely a fraction of whole population)
                               and number of individuals to be produced
    :param clustering_method: method of clustering applied in ParetoSet, str or Any clustering model
                              if str, valid options are: `affinity_propagation`, `kmeans` and `spectral`
                              if Any pass class implementing clustering interface
                              it needs cluster.fit method creating clusters in data and
                              cluster.cluster_centres_ property returning coordinates of computed cluster centres
    """

    def __init__(
        self,
        objective: Callable[[np.array], np.array],
        n_dim: int,
        mode: str,
        selection_operator: Union[str, Callable[[np.array, Any], np.array]],
        mutation_operator: Union[str, Callable[[np.array, Any], np.array]],
        crossover_operator: Union[str, Callable[[np.array, Any], np.array]],
        clustering_method: Union[str, Callable[[np.array, Any], np.array]],
    ):
        """
        Magic init method for more information see:
            >>> print(SPEAOptimizer.__doc__)
        """
        # optimization parameters
        self._objective = objective
        self._optimization_mode = mode
        self._n_dim = n_dim
        # internal variables
        self._external_set = None
        self.population = None
        self.history = None
        # operator mappings
        self._selection_operator = (
            _SELECTION_OPERATOR_MAPPING[selection_operator] if type(selection_operator) is str else selection_operator
        )
        self.mutation_operator = (
            _MUTATION_OPERATOR_MAPPING[mutation_operator] if type(mutation_operator) is str else mutation_operator
        )
        self._crossover_operator = (
            _CROSSOVER_OPERATOR_MAPPING[crossover_operator] if type(crossover_operator) is str else crossover_operator
        )
        self._clustering_method = (
            _CLUSTER_METHOD_MAPPING[clustering_method] if type(clustering_method) is str else clustering_method
        )
        self._schedules = {
            "const": const_schedule,
            "increasing_linear": increasing_linear_schedule,
            "decreasing_linear": decreasing_linear_schedule,
        }

    @property
    def pareto_front(self) -> np.array:
        """
        :return: all non dominated solutions from final generation
        """
        return self._collect_all_non_dominated_individuals(self.population)

    @property
    def population_size(self):
        """
        :return: size of current population
        """
        return self.population.shape[0]

    def optimize(
        self,
        generations: int,
        population_size: int,
        crossover_rate: float,
        mutation_rate: float,
        reducing_period: int,
        search_range: Tuple[Tuple[float, float], ...],
        mutation_strength: float = 0.1,
        clustering_parameters: dict = None,
        mutation_schedule: Union[str, Callable[[Tuple[Any, ...]], float]] = "const",
        strength_schedule: Union[str, Callable[[Tuple[Any, ...]], float]] = "const",
        crossover_schedule: Union[str, Callable[[Tuple[Any, ...]], float]] = "const",
        silent: bool = True,
        logging: bool = False,
        logging_path: Union[str, Path] = None,
    ) -> None:
        """
        Run multi objective optimization of callable function for given class instance

        :param generations: number of generations to run
        :param population_size: number of individuals in each generation
        :param crossover_rate: fraction of population taking part in crossover, must be in range <0, 1>
        :param mutation_rate: fraction of solutions mutated, must be in range <0, 1>
        :param reducing_period: number of generations to wait until reducing pareto set
        :param search_range: tuple of ranges in all dimensions
        :param mutation_strength: strength of mutation operation, should be adjusted based on search space size
        :param clustering_parameters: parameters of clustering algorithm
        :param mutation_schedule: function of mutation rate for generations,
                                  if string accepted values are `const`, `increasing_linear` and `decreasing_linear`
                                  if callable must take in three arguments and return one
        :param strength_schedule: function of mutation strength for generations
        :param crossover_schedule:  function of crossover rate for generations
        :param silent: if False, print progress bar during execution
        :param logging: if True, save population, variables and pareto set at each generation in history property
        :param logging_path: path to .csv file where logs will be saved
        """
        self.population = self._init_population(population_size, initial_search_range=search_range)
        self._external_set = ParetoSet(reducing_period, self._clustering_method, model_kwargs=clustering_parameters)

        for generation in tqdm(range(generations), disable=silent):
            # Update operator rates
            current_mutation_rate = self._set_rate(mutation_schedule, generation, mutation_rate, generations)
            current_crossover_rate = self._set_rate(crossover_schedule, generation, crossover_rate, generations)
            current_mutation_strength = self._set_rate(strength_schedule, generation, mutation_strength, generations)
            # Collect Pareto solutions
            pareto_solutions = self._collect_all_non_dominated_individuals(self.population)
            self._external_set.update(pareto_solutions)
            # Run operators
            self.population = np.concatenate([self.population, self._external_set.callback(generation)], axis=0)
            self.population = self._create_offspring(
                self.population, int(current_crossover_rate * len(self.population)), population_size
            )
            self.population = self._mutate_population(self.population, current_mutation_rate, current_mutation_strength)
            # log data
            if logging:
                self._log_data(logging_path, generation)

        return

    # private utils
    def _log_data(self, path: Union[str, Path], generation: int) -> None:
        """
        Save population to .csv file using tidy log data frame format

        :param path: path to file with logged population
        :param generation: number of current generation
        """
        logged_population = np.concatenate([self.population, self._external_set.solutions], axis=0)

        data = np.column_stack([
            flatten(logged_population),
            flatten(np.apply_along_axis(self._objective, 1, logged_population)),
            np.array([generation] * len(logged_population)),
            array_subset(logged_population, self._external_set.solutions).astype(str)
        ])

        columns_names = list(
            dims_to_column_names(flatten(logged_population), lowercase=False)
            + dims_to_column_names(flatten(np.apply_along_axis(self._objective, 1, logged_population)))
            + ["generation"]
            + ["pareto"]
        )

        logs_df = pd.DataFrame.from_records(data, columns=columns_names)
        if generation == 0:  # save header only on first generation
            logs_df.to_csv(path, index=False, header=True, mode="a")
        logs_df.to_csv(path, index=False, header=False, mode="a")

    def _set_rate(self, schedule: Any, *args):
        """
        Sets operator rate value

        :param schedule: str or Callable returning schedule values for generation
        :param args: args to schedule function

        :return: operator rate value for given generation according to its schedule
        """
        if type(schedule) == str:
            return self._schedules[schedule](*args)

        return schedule(*args)

    # Evolutionary Computation Methods
    def _init_population(self, population_size: int, initial_search_range: Tuple[Tuple[float, float], ...]) -> np.array:
        """
        Initializes population of solutions:

        Example:
            >>> self._init_population(1000, ((0, 10), (10, 20), (-10, 0)))

        Example creates population for 3D objective, with ranges [0, 10], [10, 20] and [-10, 0] in each dimension

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
        Gets all non dominated individuals from current population
        using Pareto notion of dominance

        :param population: current population
        """
        solutions = np.apply_along_axis(self._objective, 1, population)
        is_non_dominated = np.array(
            [is_non_dominated_solution(solution, solutions, mode=self._optimization_mode,) for solution in solutions]
        )

        return population[np.where(is_non_dominated == True)]

    def _mutate_population(self, population: np.array, mutation_rate: float, mutation_strength: float):
        """
        Perform mutation on randomly chosen subset of current population

        :param population: array of current solutions
        :param mutation_rate: fraction of solutions mutation will be applied to, must be in range <0, 1>
        :param mutation_strength: factor of mutation strength, should be proportional to search space size
        """
        population = np.copy(population)
        to_mutate = np.random.randint(0, len(population), int(mutation_rate * len(population)))
        population[to_mutate] = self.mutation_operator(population[to_mutate], mutation_strength)

        return population

    def _create_offspring(self, population: np.array, mating_pool_size: int, n_offspring: int) -> np.array:
        """
        Create offspring solutions using crossover

        :param population: array of current solutions
        :param mating_pool_size: number of solutions taking part in crossover
        :param n_offspring: number of solutions created by crossover
        """
        solutions = np.apply_along_axis(self._objective, 1, population)
        mating_pool = self._selection_operator(solutions, mating_pool_size, mode=self._optimization_mode)
        return vectorized_hypersphere_crossover(population[mating_pool, :], n_offspring)
