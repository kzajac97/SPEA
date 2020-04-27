import numpy as np
from numba import jit

# util constants
X1_INDEX = 0
X2_INDEX = 1
# objective function constants
CONST_A_1 = 0.5*np.sin(1.0) - 2.0*np.cos(1.0) + 1.0*np.sin(2.0) - 1.5*np.cos(2.0)
CONST_A_2 = 1.5*np.sin(1.0) - 1.0*np.cos(1.0) + 2.0*np.sin(2.0) - 0.5*np.cos(2.0)


@jit
def schaffer_objective_n1(x: np.array) -> np.array:
    """
    Schaffer test functions for multi objective optimization
    Input D: 1
    Output D: 2
    Search Range: [-10e5, 10e5]
    """
    return np.array([np.power(x, 2),
                     np.power((x - 2), 2)])


@jit
def schaffer_objective_n2(x: np.array) -> np.array:
    """
    Schaffer test functions for multi objective optimization
    Input D: 1
    Output D: 2
    Search Range: [-5, 10]
    """
    return np.array([
        -x * (x <= 1) + (x - 2) * (1 < x <= 3) + (4 - x) * (3 < x <= 4) + (x - 4) * (x > 4),
        np.power((x - 5), 2)
    ])


@jit
def _b1_term(x, y):
    """
    Term function for `polonis_objective`
    """
    return 0.5*np.sin(x) - 2.0*np.cos(x) + 1.0*np.sin(y) - 1.5*np.cos(y)


@jit
def _b2_term(x, y):
    """
    Term function for `polonis_objective`
    """
    return 1.5*np.sin(x) - 1.0*np.cos(x) + 2.0*np.sin(y) - 0.5*np.cos(y)


@jit
def polonis_objective(values: np.array) -> np.array:
    """
    Polonis test functions for multi objective optimization
    Input D: 2
    Output D: 2
    Search Range: x in [-PI, INF)
                  y in (-INF, PI]
    """
    x, y = values
    return np.array([
        1 + np.power(CONST_A_1 - _b1_term(x, y), 2) + np.power(CONST_A_2 - _b2_term(x, y), 2),
        np.power(x + 3, 2) + np.power(y + 1, 2)
    ])


@jit
def _f1_function(values):
    """
    Term function for `zietler_deb_thiele_objective_n1`
    """
    return 1 - np.exp(-4 * values[X1_INDEX]) * np.power(np.sin(6 * np.pi * values[X1_INDEX]), 6)


@jit
def _g1_function(values):
    """
    Term function for `zietler_deb_thiele_objective_n1`
    """
    return 1 + (9/29) * np.sum(values[X2_INDEX:])


@jit
def _g4_function(values):
    """
    Term function for `zietler_deb_thiele_objective_n1`
    """
    return 91 + np.sum(np.power(values[X2_INDEX:], 2) - 10*np.cos(4*np.pi*values[X2_INDEX:]))


@jit
def _g6_function(values):
    """
    Term function for `zietler_deb_thiele_objective_n1`
    """
    return 1 + 9*np.power((1/9)*np.sum(values[X2_INDEX:]), 0.25)


@jit
def _h1_function(f_value, g_value):
    """
    Term function for `zietler_deb_thiele_objective_n1`
    """
    return 1 - np.sqrt(f_value / g_value)


@jit
def _h2_function(f_value, g_value):
    """
    Term function for `zietler_deb_thiele_objective_n1`
    """
    return 1 - np.power(f_value / g_value, 2)


@jit
def _h3_function(f_value, g_value):
    """
    Term function for `zietler_deb_thiele_objective_n1`
    """
    return 1 - 1 - np.sqrt(f_value / g_value) - (f_value / g_value) * np.sin(10*np.pi*f_value)


@jit
def zietler_deb_thiele_objective_n1(values: np.array) -> np.array:
    """
    Zietler-Deb-Thiele test functions for multi objective optimization
    Input D: 30
    Output D: 2
    Search Range: [-1, 1]
    """
    return np.array([
        values[X1_INDEX],
        _g1_function(values) * _h1_function(values[X1_INDEX], _g1_function(values))
    ])


@jit
def zietler_deb_thiele_objective_n2(values: np.array) -> np.array:
    """
    Zietler-Deb-Thiele test functions for multi objective optimization
    Input D: 30
    Output D: 2
    Search Range: [-1, 1]
    """
    return np.array([
        values[X1_INDEX],
        _g1_function(values) * _h2_function(values[X1_INDEX], _g1_function(values))
    ])


@jit
def zietler_deb_thiele_objective_n3(values: np.array) -> np.array:
    """
    Zietler-Deb-Thiele test functions for multi objective optimization
    Input D: 30
    Output D: 2
    Search Range: [-1, 1]
    """
    return np.array([
        values[X1_INDEX],
        _g1_function(values) * _h3_function(values[X1_INDEX], _g1_function(values))
    ])


@jit
def zietler_deb_thiele_objective_n4(values: np.array) -> np.array:
    """
    Zietler-Deb-Thiele test functions for multi objective optimization
    Input D: 10
    Output D: 2
    Search Range: x1 in [0, 1]
                  x2 in [-5, 5]
    """
    return np.array([
        values[X1_INDEX],
        _g4_function(values) * _h1_function(values[X1_INDEX], _g4_function(values))
    ])


@jit
def zietler_deb_thiele_objective_n6(values: np.array) -> np.array:
    """
    Zietler-Deb-Thiele test functions for multi objective optimization
    Input D: 10
    Output D: 2
    Search Range: x in [0, 1]
    """
    return np.array([
        _f1_function(values),
        _g6_function(values) * _h2_function(_f1_function(values), _g6_function(values))
    ])


@jit
def viennet_objective(values: np.array) -> np.array:
    """
    Viennet test functions for multi objective optimization
    Input D: 2
    Output D: 3
    Search Range: x in [-3, 3]
    """
    x, y = values
    return np.array([
        0.5*(np.power(x, 2) + np.power(y, 2)) + np.sin(np.power(x, 2) + np.power(y, 2)),
        15 + (1/8)*np.power(3*x - 2*y + 4, 2) + (1/27)*np.power(x - y + 1, 2),
        1 / (np.power(x, 2) + np.power(y, 2) + 1) - 1.1*np.exp(-1*(np.power(x, 2) + np.power(y, 2)))
    ])
