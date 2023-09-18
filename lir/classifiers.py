import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator


def TLM_calc_MSwithin(X, y):
    """
    X np.array of measurements, rows are objects, columns are variables
    y np 1d-array of labels. labels from {1, ..., n} with n the number of objects. Repetitions get the same label.
    returns: mean within covariance matrix, np.array
    """
    # use pandas functionality to allow easy calculation
    df = pd.DataFrame(X, index=pd.Index(y, name="label"))
    # filter out single-repetitions,since they do not contribute to covariance calculations
    grouped = df.groupby(axis='index', by='label')
    filtered = grouped.filter(lambda x: x[0].count() > 1)
    # make groups again by windownr and calculate covariance matrices per window
    grouped = filtered.groupby(axis='index', by='label')
    covars = grouped.cov(ddof=1)
    # add index names to allow grouping by element, group by element and get mean covariance matrix
    covars.index.names = ["Source", "Variable"]
    grouped_by_element = covars.groupby(["Variable"])
    mean_covars = grouped_by_element.mean()
    return np.array(mean_covars)

def TLM_calc_means(X, y):
    """
    X np.array of measurements, rows are objects, columns are variables
    y np 1d-array of labels. labels from {1, ..., n} with n the number of objects. Repetitions get the same label.
    returns: means per object, rows sorted from {1, ..., n}, n * (number of variables) np.array
    """
    # use pandas functionality to allow easy calculation
    df = pd.DataFrame(X, index=pd.Index(y, name="label"))
    # filter out single-repetitions,since they do not contribute to covariance calculations
    grouped = df.groupby(axis='index', by='label')
    means = grouped.mean()
    return np.array(means)


def TLM_calc_kappa(X, y):
    """
    X np.array of measurements, rows are objects, columns are variables
    y np 1d-array of labels. labels from {1, ..., n} with n the number of objects. Repetitions get the same label.
    returns: kappa (float). Kappa is used in cacluation of T0, where T0 is between variance of means. Kappa converts
        variance at measurement level to variance at means level.
    """
    # use pandas functionality to allow easy calculation
    df = pd.DataFrame(X, index=pd.Index(y, name="label"))
    # group per source
    grouped = df.groupby(axis='index', by='label')
    # get the number of repetitions per source
    reps = grouped.size()
    # calculate sum_reps_sq and kappa
    sum_reps_sq = sum([rep ** 2 for rep in reps])
    kappa = (reps.sum() - sum_reps_sq/reps.sum())/(len(reps)-1)
    return kappa