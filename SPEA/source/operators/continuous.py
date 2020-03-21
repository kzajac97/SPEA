import numpy as np
from numba import jit


@jit
def crossover(individuals: np.array) -> np.array:
    """
    Crossover between two individual in floating point gene representation
    Function in compiled just in time, because it is called many times during algorithm execution

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
    Function in compiled just in time, because it is called many times during algorithm execution

    Mutation works by drawing N samples from random normal distribution add adding in to individual

    :param individual: single solution represented as numpy array with shape (N_OBJECTIVE_DIMS, )
    :param mutation_strength: strength of mutation operation, should be adjusted based on search space size

    :return: Mutated individual solution
    """
    offset = np.random.randn(individual.shape[0])
    return mutation_strength * offset + individual


@jit
def selection(population: np.array) -> np.array:
    ...

