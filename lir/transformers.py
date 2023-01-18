from typing import Callable, Optional

import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.stats import rankdata
import warnings

from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from lir.util import to_log_odds


class EstimatorTransformer(TransformerMixin):
    """
    A wrapper for an estimator to make it behave like a transformer.

    In particular, it implements `transform` by calling `predict_proba` on the underlying estimator, and transforming
    the probabilities to their corresponding log odds value. Optionally, an alternative transformation function can be
    specified.
    """
    def __init__(self, estimator, transform_probabilities: Optional[Callable] = to_log_odds):
        self.estimator = estimator
        self.transform_probabilities = transform_probabilities

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        return self.transform_probabilities(self.estimator.predict_proba(X)[:, 1])

    def __getattr__(self, item):
        return getattr(self.estimator, item)


class ComparisonFunctionTransformer(FunctionTransformer):
    """
    A wrapper for a distance function to make it behave like a transformer.

    This is essentially a `FunctionTransformer` except that it expects a function takes two arguments of the same
    dimensions: `X` and `Y`, where each row in `X` is compared to the row with the same index in `Y`.

    See: the `paired_*` distance functions in `sklearn.metrics.pairwise`.
    """
    def __init__(self, distance_function: Callable):
        self.distance_function = distance_function
        super().__init__(self.transformer_function)

    def transformer_function(self, X):
        if len(X.shape) == 2:
            return self.distance_function(X)
        elif len(X.shape) == 3:
            vectors = [X[:, :, i] for i in range(X.shape[2])]
            return self.distance_function(*vectors)
        else:
            raise ValueError(f"unexpected input shape; expected: (*, *) or (*, *, *); found: {X.shape}")


class AbsDiffTransformer(sklearn.base.TransformerMixin):
    """
    Takes an array of sample pairs and returns the element-wise absolute difference.

    Expects:
        - X is of shape (n,f,2) with n=number of pairs; f=number of features; 2=number of samples per pair;
    Returns:
        - X has shape (n, f)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[2] == 2

        return np.abs(X[:,:,0] - X[:,:,1])


class PercentileRankTransformer(sklearn.base.TransformerMixin):
    """
    Compute the percentile rankings of a dataset, relative to another dataset.
    Rankings are in range [0, 1]. Handling ties: the maximum of the ranks that
    would have been assigned to all the tied values is assigned to each value.

    To be able to compute the rankings of dataset Z relative to dataset X,
    'fit' will create a ranking function for each feature, based on X.
    'transform' will apply ranking of Z based on dataset X.

    Fit:
    Expects:
        - X is of shape (n, f) with n = number of measurements,
        f = number of features

    Transform:
    Expects:
        - X is of shape (m, f) with m = number of measurements,
        f = number of features
    Returns:
        - rankings with shape (m, f)
    """
    def __init__(self):
        self.rank_functions = None

    def fit(self, X, y=None):
        assert len(X.shape) == 2
        ranks_X = rankdata(X, method='max', axis=0)/len(X)
        # if a feature is a constant int value, interp1d returns nan -> convert to float
        X = X.astype(float)
        self.rank_functions = [interp1d(X[:, i], ranks_X[:, i], bounds_error=False,
                                        fill_value=(0, 1)) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        assert self.rank_functions, "transform() called before fit()"
        assert len(X.shape) == 2
        assert X.shape[1] == len(self.rank_functions),\
            "number of features used for fit() and transform() should be equal"
        ranks = [self.rank_functions[i](X[:, i]) for i in range(X.shape[1])]
        return np.stack(ranks, axis=1)


class InstancePairing(sklearn.base.TransformerMixin):
    def __init__(self,
                 same_source_limit=None,
                 different_source_limit=None,
                 ratio_limit=None,
                 seed=None):
        """
        Creates pairs of instances.

        This transformer takes `X` and `y` as input, and returns `X_paired` and `y_paired`.
        `X_paired` is the carthesian product of `X` and itself.
        In other words, `X_paired` contains all possible pairs of feature vectors in `X`, meaning that
        `X_paired.shape[0] == X.shape[0]*X.shape[0]`, except if their number exceeds the values of parameters
        `same_source_limit` and `different_source_limit`, in which case pairs are randomly drawn from the full set of
        pairs. Also, a maximum ratio between the same and different source pairs can be specified with 'ratio_limit'.

        Note that this transformer may cause performance problems with large datasets,
        even if the number of instances in the output is limited.

        Parameters:
            - same_source_limit (int or None): the maximum number of same source pairs (None = no limit)
            - different_source_limit (int or None or 'balanced'): the maximum number of different source pairs (None = no limit; 'balanced' = number of same source pairs)
            - ratio_limit (int or None): maximum ratio between same source and different source pairs.
                Ratio = ds pairs / ss pairs. The number of ds pairs will not exceed ratio_limit * ss pairs.
                If both ratio and same_source_limit/different_source_limit are specified,
                the number of pairs is chosen such that the ratio_limit is preserved and
                the limit(s) are not exceeded, while taking as many pairs as possible within these constraints.
            - seed (int or None): seed to make pairing reproducible
        """
        self._ss_limit = same_source_limit
        self._ds_limit = different_source_limit
        self._ratio_limit = ratio_limit
        self.rng = np.random.default_rng(seed=seed)

        if self._ds_limit == 'balanced':
            warnings.warn('The argument \'balanced\' is deprecated. '
                          'Use ratio_limit instead.', DeprecationWarning, stacklevel=2)
            self._ds_limit = None
            self._ratio_limit = 1

    def fit(self, X):
        return self

    def transform(self, X, y):
        """
        Expects:
            - X is of shape (n,f) with n=number of samples; f=number of features
            - y is of shape (n,)
        Returns:
            - X_paired, y_paired: X_paired has shape (m,f,2); y_paired has shape (m,); with m=number of pairs.
                For example, `(X_paired[a,:,0], X_paired[a,:,1])` is the a^th pair of feature vectors, and `y_paired[a]`
                is 1 if the labels of both vectors in `y` are equal (same source), or 0 otherwise (different source).
        Attributes:
            - self.pairing: an array of shape (m,2) with indexes of X input array which contribute to the pairs
        """
        pairing = np.array(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[0]))).T.reshape(-1, 2)  # generate all possible pairs
        same_source = y[pairing[:, 0]] == y[pairing[:, 1]]

        rows_same = np.where((pairing[:, 0] < pairing[:, 1]) & same_source)[0]  # pairs with different id and same source
        rows_diff = np.where((pairing[:, 0] < pairing[:, 1]) & ~same_source)[0]  # pairs with different id and different source

        if self._ss_limit is not None and rows_same.size > self._ss_limit:
            rows_same = self.rng.choice(rows_same, self._ss_limit, replace=False)

        n_ds_pairs = min(x for x in [rows_same.size * self._ratio_limit if self._ratio_limit else None,
                                     self._ds_limit,
                                     rows_diff.size
                                     ] if x is not None)

        if n_ds_pairs < rows_diff.size:
            rows_diff = self.rng.choice(rows_diff, n_ds_pairs, replace=False)

        pairing = np.concatenate([pairing[rows_same, :], pairing[rows_diff, :]])
        self.pairing = pairing

        X = np.stack([X[pairing[:, 0]], X[pairing[:, 1]]], axis=2)  # pair instances by adding another dimension
        y = np.concatenate([np.ones(rows_same.size), np.zeros(rows_diff.size)])  # apply the new labels: 1=same_source versus 0=different_source

        return X, y
