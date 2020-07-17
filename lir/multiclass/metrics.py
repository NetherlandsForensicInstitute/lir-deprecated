import numpy as np
import scipy.stats

from .util import scores_Xy_to_Xn, get_classes_from_scores_Xy


def cllr(lrs):
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood ratios. The LRs must be formatted as
    the ratio of the probability of the correct class versus the probability of the wrong class.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    Parameters
    ----------
    lrs : a numpy array of LRs

    Returns
    -------
    cllr
        the log likelihood ratio cost
    """
    with np.errstate(divide='ignore'):
        return np.mean(np.log2(1 + 1/lrs))


def geometric_mean(values):
    """
    Calculates the geometric mean over all values in an array.
    """
    return scipy.stats.mstats.gmean(values)


def by_class(metric, scores, y, classes=None):
    """
    Calculates the average of a metric for each class separately.

    Same as `macro()` except that it returns a value for each class separately instead of averaging the values.
    """
    with np.errstate(divide='ignore'):
        lrs_by_class = scores_Xy_to_Xn(scores, y, classes)
    assert len(lrs_by_class) == scores.shape[1], 'number of classes in `y` does not match the number of columns in `lrs`'
    metric_by_class = np.array([metric(lrs_cls[:,i]) for i, lrs_cls in enumerate(lrs_by_class)])
    return metric_by_class


def macro(metric, scores, y, classes=None):
    """
    Calculates the macro average of a metric with equal class weights.

    Parameters
    ----------
    metric: a function which calculates the metric

    scores : a numpy array of scores; rows are scores by sample; columns are scores by class

    y: a numpy array of classes; each element corresponds to a row in `scores`

    Returns
    -------
    macro average
        the macro average of the metric
    """
    metric_by_class = by_class(metric, scores, y, classes)
    return np.mean(metric_by_class)


def micro(metric, scores, y, classes=None):
    """
    Calculates the micro average of a metric with equal class weights.

    Same as `macro()` except that it returns the micro average instead of the macro average.
    """
    classes = get_classes_from_scores_Xy(scores, y, classes)
    assert np.all(np.arange(scores.shape[1]) == classes), 'classes must be numbered 0..n and each class must occur at least once'

    lrs = scores[np.arange(scores.shape[0]), y]
    return metric(lrs)
