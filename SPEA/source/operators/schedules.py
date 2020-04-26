import numpy as np
from numba import jit


"""
Functions returning schedules for changing operators rate in SPEA algorithm
All follow the same signature:
    :param generation: current generation index
    :param factor: initial or final operator rate
    :param n_generations: number of generations to run 
    
    :return: float with value of parameter rate for current generation
"""
@jit
def const_schedule(generation: int, factor: float, n_generations: int) -> float:
    return factor


@jit
def increasing_linear_schedule(generation: int, factor: float, n_generations: int) -> float:
    return (factor/n_generations) * generation


@jit
def decreasing_linear_schedule(generation: int, factor: float, n_generations: int) -> float:
    return factor - (factor/n_generations) * generation
