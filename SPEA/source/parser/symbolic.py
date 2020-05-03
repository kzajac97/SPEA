import string
import re
from typing import Union

import numpy as np

from source.parser.expression import Expression

X_ALPHABETICAL_INDEX = 23

_VARIABLE_ORDERING_KEYS = {
    # X Y Z are first 3 variables in mathematical ordering
    "mathematical": string.ascii_uppercase[X_ALPHABETICAL_INDEX:] + string.ascii_uppercase[:X_ALPHABETICAL_INDEX - 1],
    "alphabetic": string.ascii_uppercase,
}


class Parser:
    """
    Parser for multi dimensional symbolic vector functions

    :example:
        >>> parser = Parser(["2*X + sin(Y)", "5*Y + log(Z)", "exp(X)", "-1*Z / X"], ordering="mathematical")
        >>> parser([1, 1, 1])
        ... array([2.8, 5.0, 2.7, -1.0], dtype=object)

    :warning: variables must be uppercase and multiplication is required
    """
    def __init__(self, expressions: list, ordering: Union[str, list] = "mathematical"):
        """
        :param expressions: list of symbolic expressions
        :param ordering: variable ordering, can be str to choose from defaults
                         or list of string for custom ordering
        """
        self._variable_ordering = _VARIABLE_ORDERING_KEYS[ordering] if type(ordering) is str else ordering
        self._variables = self._get_all_variables(expressions)
        self._expressions = [Expression(expression, self._variables) for expression in expressions]

    def _get_all_variables(self, expressions: list):
        """
        :return: list of all variables found in expressions
        """
        variables = []
        for expression in expressions:
            variables.extend(re.findall(r"[A-Z]", expression))

        return sorted(list(set(variables)), key=lambda item: self._variable_ordering.index(item))

    def __call__(self, values):
        """
        :param values: values at which to evaluate symbolic expression
        """
        return np.array([expression(values) for expression in self._expressions])
