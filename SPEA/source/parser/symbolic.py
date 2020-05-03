import string
import re
from typing import List, Union

from source.parser.expression import Expression, VectorExpression

X_ALPHABETICAL_INDEX = 23

_VARIABLE_ORDERING_KEYS = {
    # X Y Z are first 3 variables in mathematical ordering
    "mathematical": string.ascii_uppercase[X_ALPHABETICAL_INDEX:] + string.ascii_uppercase[:X_ALPHABETICAL_INDEX - 1],
    "alphabetic": string.ascii_uppercase,
}


def _get_all_variables(expressions: list, ordering: Union[str, list]):
    """
    :return: list of all variables found in expressions
    """
    variables = []
    for expression in expressions:
        variables.extend(re.findall(r"[A-Z]", expression))

    return sorted(list(set(variables)), key=lambda item: ordering.index(item))


def parse_expression(expression: str) -> Expression:
    """
    :example:
        >>> equation = parse_expression("2*X + sin(Y) + exp(Z)")
        >>> equation([1, 1, 1])
        ... 5.56
    :return:
    """
    return Expression(expression)


def parse_vector_expression(
        functions: List[str], ordering: Union[str, list] = "mathematical"
) -> VectorExpression:
    """
    :example:
        >>> vector_expression = parse_vector_expression(
        ...  ["2*X + sin(Y)",
        ...   "5*Y + log(Z)",
        ...   "exp(X)",
        ...   "-1*Z / X"],
        ... ordering="mathematical")
        >>> vector_expression([1, 1, 1])
        ... array([2.8, 5.0, 2.7, -1.0], dtype=object)

    :param functions: list of vector functions
    :param ordering: variable ordering, can be str to choose from defaults
                     or list of string for custom ordering

    :return: callable VectorExpression object
    """
    ordering = _VARIABLE_ORDERING_KEYS[ordering] if type(ordering) is str else ordering
    variables = _get_all_variables(functions, ordering)
    return VectorExpression(expressions=functions, variables=variables, ordering=ordering)