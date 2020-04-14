from typing import Any

import numpy as np


class ParetoSet:
    """
    Class holds implementation of external Pareto set used by SPEA algorithm
    """
    def __init__(self, reducing_period: int, model: Any, model_kwargs: dict = None):
        """
        :param reducing_period: number of generations to wait for reducing Pareto Set
        :param model: clustering model
        :param model_kwargs: parameters to clustering model, see:
        https://scikit-learn.org/stable/modules/clustering.html
        """
        self._model = model(**model_kwargs)
        self._reducing_period = reducing_period
        self.solutions = None

    def update(self, collected_solutions: np.array) -> None:
        """
        Update Pareto set with new solutions
        """
        if self.solutions is None:  # on first epoch
            self.solutions = collected_solutions

        self.solutions = np.concatenate([self.solutions, collected_solutions])

    def _reduce(self):
        """
        Run clustering algorithm on Pareto set to reduce it
        """
        self._model.fit(self.solutions)
        self.solutions = np.copy(self._model.cluster_centers_)

    def callback(self, n_generation: int) -> np.array:
        """
        Function called at each epoch in optimization loop
        """
        if n_generation % self._reducing_period == 0:
            self._reduce()

        return self.solutions
