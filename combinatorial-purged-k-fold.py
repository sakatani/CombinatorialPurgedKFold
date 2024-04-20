import itertools as itt

import numpy as np


class CombinatorialPurgedKFold():
    """
    Combinatorial Purged K-Fold
    Reference: Advances in financial machine learning, Marcos López de Prado, 2018.
    """
    def __init__(
        self,
        n_ticks: int,
        n_folds: int,
        n_test_folds: int,
        embargo_days: int | list[int],
        verbose: bool = True
    ) -> None:
        """
        Args:
            n_ticks (int): Number of ticks.
            n_folds (int): Number of folds.
            n_test_folds (int): Num of folds used for test.
            embargo_days (int | list[int]): Length of embargo days.
                List argument assigns different lengths to preceding and following days of test folds. 
            verbose (bool): if resulted number of simulations and paths are displayed.3333333
        """
        self.n_ticks = n_ticks
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        if isinstance(embargo_days, int):
            self.pre_embargo_days = embargo_days
            self.post_embargo_days = embargo_days
        elif isinstance(embargo_days, list):
            self.pre_embargo_days = embargo_days[0]
            self.post_embargo_days = embargo_days[1]
        self.verbose = verbose

        # split data into n_folds, with n_folds << n_ticks
        # this will assign each index position to a fold position
        self.fold_map = np.arange(self.n_ticks) // (self.n_ticks // self.n_folds)
        self.fold_map[self.fold_map == self.n_folds] = self.n_folds - 1

        # generate the combinations
        self.test_fold_combs = np.array(
            list(itt.combinations(np.arange(self.n_folds), self.n_test_folds))
        ).reshape(-1, self.n_test_folds)
        self.n_simulations = len(self.test_fold_combs)
        self.n_paths = self.n_simulations * self.n_test_folds // self.n_folds

        # test_map is a T x C(n_folds, n_test_folds) array
        # where each column is a logical array
        # indicating which observation is in the test set
        self.test_fold_map = np.full((self.n_folds, self.n_simulations), fill_value=False)
        self.preceding_embargo_fold_map = np.full((self.n_folds, self.n_simulations), fill_value=False)
        self.following_embargo_fold_map = np.full((self.n_folds, self.n_simulations), fill_value=False)

    def _embargo_folds(
        self,
        test_fold_idx: list[int]
    ) -> tuple[list[int], list[int]]:
        """Calculate indices of folds including embargoed data
        Args:
            test_folds_idx (list[int]): Indices of test folds
        Returns:
            tuple[list[int], list[int]]: 
                Fold indices including a embargo period preceding the test fold,
                Fold indices including a embargo period following the test fold
        """
        preceding_embargo_fold_idx = []
        following_embargo_fold_idx = []

        for g in test_fold_idx:
            if g > 0:
                preceding_embargo_fold_idx.append(g - 1)
            if g < self.n_folds - 1:
                following_embargo_fold_idx.append(g + 1)
        preceding_embargo_fold_idx = list(set(preceding_embargo_fold_idx) - set(test_fold_idx))
        following_embargo_fold_idx = list(set(following_embargo_fold_idx) - set(test_fold_idx))

        return preceding_embargo_fold_idx, following_embargo_fold_idx

    def __call__(self) -> tuple[list[list[bool]], list[list[bool]]]:
        """
        Returns:
            tuple[list[list[bool]], list[list[bool]]]: 
                Mappping ticks to whether they are in a test fold or not,
                Mapping ticks to whether they are embargoed data or not
        """
        test_map = np.full((self.n_ticks, self.n_simulations), fill_value=False)
        embargo_map = np.full((self.n_ticks, self.n_simulations), fill_value=False)

        # assign test folds for each of the C(n_folds, n_test_folds) simulations
        for k, pair in enumerate(self.test_fold_combs):
            i, j = pair
            self.test_fold_map[[i, j], k] = True

            preceding_embargo_fold_idx, following_embargo_fold_idx = self._embargo_folds([i, j])

            self.preceding_embargo_fold_map[preceding_embargo_fold_idx, k] = True
            self.following_embargo_fold_map[following_embargo_fold_idx, k] = True

            # assigning the test folds
            mask = (self.fold_map == i) | (self.fold_map == j)
            test_map[mask, k] = True

            # assigning the embargo data
            for idx in preceding_embargo_fold_idx:
                mask = np.where(self.fold_map == idx)[0][-self.pre_embargo_days:]
                embargo_map[mask, k] = True

            for idx in following_embargo_fold_idx:
                mask = np.where(self.fold_map == idx)[0][:self.post_embargo_days]
                embargo_map[mask, k] = True

        if self.verbose:
            print('Num Simulation:', self.n_simulations)
            print('Num Paths:', self.n_paths)

        return test_map, embargo_map
