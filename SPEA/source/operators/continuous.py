import numpy as np
from numba import jit


@jit
def crossover(individuals: np.array) -> np.array:
    """
    Crossover between two individual in floating point gene representation

    Crossover work by finding a vector between to individual point, computes distance between there points and than
    shifts in by random amount from one individual in the direction of the other individual

    :param individuals: array of two individual solution selected from mating pool

    :return: single solution being crossover of two parent solutions
    """
    x, y = individuals

    offset_vector = y - x  # vector from one individual in direction on second individual
    translation_length = np.random.normal(0.5, 0.25)  # scaled Gaussian, draws samples between [0, 1] centered at 0.5
    return (translation_length * offset_vector) + x


@jit
def mutation(individual: np.array, mutation_strength: float = 0.1) -> np.array:
    """
    Performs Gaussian mutation operation on single individual in floating point gene representation,
    Mutation works by drawing N samples from random normal distribution add adding in to individual

    :param individual: single solution represented as numpy array with shape (N_OBJECTIVE_DIMS, )
    :param mutation_strength: strength of mutation operation, should be adjusted based on search space size

    :return: Mutated individual solution
    """
    offset = np.random.randn(individual.shape[0])
    return mutation_strength * offset + individual


def vectorized_crossover(mating_pool: np.array, n_offspring: int) -> np.array:
    """
    Crossover between two individuals applied to whole population in floating point gene representation

    Crossover work by finding a vector between to individual point, computes distance between there points and than
    shifts in by random amount from one individual in the direction of the other individual

    :param mating_pool: array of solutions, chosen to mate in shape (N_INDIVIDUALS, 2, N_OBJECTIVE_DIMS, )
    :param n_offspring: number of solutions produces via crossover

    :return: population array after applying crossover, with shape (N_INDIVIDUALS, N_OBJECTIVE_DIMS, )
    """
    mating_pool = mating_pool[np.random.randint(0, mating_pool.shape[0], n_offspring * 2)]
    mating_pool = mating_pool.reshape(n_offspring, 2, mating_pool.shape[-1])

    x = mating_pool[:, 0, :]
    y = mating_pool[:, 1, :]

    offsets = y - x
    lengths = np.random.normal(0.5, 0.25, n_offspring)
    return np.apply_along_axis(np.multiply, 0, offsets, lengths) + x


def vectorized_mutation(population: np.array, mutation_strength: float = 0.1) -> np.array:
    """
    Performs Gaussian mutation operation on population of individuals in floating point gene representation,
    Mutation works by drawing N samples from random normal distribution add adding in to individual

    :param population: array of solutions, with shape (N_INDIVIDUALS, N_OBJECTIVE_DIMS, )
    :param mutation_strength: strength of mutation operation, should be adjusted based on search space size

    :return: array containing mutated population
    """
    offset = np.random.randn(population.shape[0] * population.shape[1]).reshape(population.shape)
    return mutation_strength * offset + population
