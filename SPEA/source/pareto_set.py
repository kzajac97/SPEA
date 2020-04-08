import numpy as np
from sklearn.cluster import AffinityPropagation


class ParetoSet:
    """
    Class holds implementation of external Pareto set used by SPEA algorithm
    """

    def __init__(self, reducing_period: int, model_kwargs: dict = None):
        """
        :param reducing_period:
        :param model_kwargs: parameters to AffinityPropagation model, see:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
        """
        self._model = AffinityPropagation(**model_kwargs)
        self._reducing_period = reducing_period
        self.solutions = np.array([])

    def update(self, collected_solutions: np.array) -> None:
        """
        Update Pareto set with new solutions
        """
        self.solutions = np.concatenate([self.solutions, collected_solutions])

    def _reduce(self):
        """
        Run clustering algorithm on Pareto set to reduce it
        """
        self._model.fit(self.solutions)
        self.solutions = self._model.cluster_centers_

    def callback(self, epoch: int) -> np.array:
        """
        Function called at each epoch in optimization loop
        """
        if epoch % self._reducing_period == 0:
            self._reduce()

        return self.solutions
