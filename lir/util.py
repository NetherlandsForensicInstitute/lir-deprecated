import collections
import inspect
import numpy as np
import warnings
from functools import partial

LR = collections.namedtuple('LR', ['lr', 'p0', 'p1'])


def get_classes_from_Xy(X, y, classes=None):
    assert len(X.shape) >= 1, f'expected: X has at least 1 dimensions; found: {len(X.shape)} dimensions'
    assert len(y.shape) == 1, f'expected: y is a 1-dimensional array; found: {len(y.shape)} dimensions'
    assert X.shape[0] == y.size, f'dimensions of X and y do not match; found: {X.shape[0]} != {y.size}'

    return np.unique(y) if classes is None else np.asarray(classes)


def Xn_to_Xy(*Xn):
    """
    Convert Xn to Xy format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class."""
    Xn = [np.asarray(X) for X in Xn]
    X = np.concatenate(Xn)
    y = np.concatenate([np.ones((X.shape[0],), dtype=np.int8) * i for i, X in enumerate(Xn)])
    return X, y


def Xy_to_Xn(X, y, classes=[0, 1]):
    """
    Convert Xy to Xn format.

    Xn is a format where samples are divided into separate variables based on class.
    Xy is a format where all samples are concatenated, with an equal length variable y indicating class."""

    classes = get_classes_from_Xy(X, y, classes)
    return [X[y == yvalue] for yvalue in classes]


def to_probability(odds):
    """
    Converts odds to a probability

    Returns
    -------
       1                , for odds values of inf
       odds / (1 + odds), otherwise
    """
    inf_values = odds == np.inf
    with np.errstate(invalid='ignore'):
        p = np.divide(odds, (1 + odds))
    p[inf_values] = 1
    return p


def to_odds(p):
    """
    Converts a probability to odds
    """
    with np.errstate(divide='ignore'):
        return p / (1 - p)


def to_log_odds(p):
    """
    Converts probability values to their log odds with base 10.
    """
    with np.errstate(divide='ignore'):
        complement = 1 - p
        return np.log10(p) - np.log10(complement)


def from_log_odds_to_probability(log_odds):
    return to_probability(10 ** log_odds)


def ln_to_log10(ln_data):
    return np.log10(np.e) * ln_data


def warn_deprecated():
    warnings.warn(f'the function `{inspect.stack()[1].function}` is no longer maintained; please check documentation for alternatives')


class Bind(partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder.
    Can be used to fix parameters not at the end of the list of parameters (which is a limitation of partial).
    """

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)
