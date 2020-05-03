import re

import numpy as np
import sympy


class Expression:
    """

    """
    def __init__(self, symbolic_expression: str):
        """
        :param symbolic_expression: string with symbolically written equation
        """
        self._expression = symbolic_expression
        self._variables = sorted(re.findall("[A-Z]", symbolic_expression))

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
