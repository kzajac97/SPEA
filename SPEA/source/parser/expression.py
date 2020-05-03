import re
from typing import Union

import numpy as np
import sympy


class Expression:
    """
    Callable class for parsing symbolic expressions using sympy syntax

    :warning: variables must be uppercase and multiplication is required
    """
    def __init__(self, symbolic_expression: str, variables: list = None):
        """
        :param symbolic_expression: string with symbolically written equation
        """
        self._expression = symbolic_expression
        self._variables = variables if variables else sorted(re.findall("[A-Z]", symbolic_expression))

    def _get_translation_dict(self, values: list) -> dict:
        """
        :return: create string translation dictionary for variable names and passed values
        """
        return {variable: str(value) for variable, value in zip(self._variables, values)}

    def __call__(self, values: np.array) -> float:
        """
        :param values: values at which to evaluate symbolic expression
        """
        return sympy.N(self._expression.translate(str.maketrans(self._get_translation_dict(values))))


class VectorExpression:
    """
    Object holding parsed multi dimensional symbolic vector functions

    :warning: variables must be uppercase and multiplication is required
    """
    def __init__(self, expressions: list, variables: list, ordering: Union[str, list]):
        """
        :param expressions: list of symbolic expressions
        :param ordering: variable ordering, can be str to choose from defaults
                         or list of string for custom ordering
        """
        self._variable_ordering = ordering
        self._variables = variables
        self._expressions = [Expression(expression, self._variables) for expression in expressions]

    def __call__(self, values):
        """
        :param values: values at which to evaluate symbolic expression
        """
        return np.array([expression(values) for expression in self._expressions])
