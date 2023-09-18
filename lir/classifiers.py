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


def TLM_calc_T0(X, y):
    """
    X np.array of measurements, rows are objects, columns are variables
    y np 1d-array of labels. labels from {1, ..., n} with n the number of objects. Repetitions get the same label.
    returns: kappa (float). Kappa is used in cacluation of T0, where T0 is between variance of means. Kappa converts
        variance at measurement level to variance at means level.
    """
    # calculate MSwithin
    MSwithin = TLM_calc_MSwithin(X, y)

    # use pandas functionality to allow easy calculation
    df = pd.DataFrame(X, index=pd.Index(y, name="label"))
    # group per source
    grouped = df.groupby(axis='index', by='label')
    # get the number of repetitions per source
    reps = np.array(grouped.size()).reshape((-1,1))
    # calculate sum_reps_sq and kappa
    sum_reps_sq = sum(reps**2)
    n_sources = len(reps)
    kappa = (reps.sum() - sum_reps_sq/reps.sum())/(n_sources-1)

    # calculate grand mean
    grand_mean = np.array(df.mean()).reshape((1,-1))
    # calculate mean per source
    means = TLM_calc_means(X, y)
    # calculate squared differences
    differences = (means - grand_mean)
    n_variables = differences.shape[1]

    differences = differences.reshape((n_variables, 1, n_variables))
    differences_t = differences.reshape((1, n_variables, n_variables))
    result = np.matmul(differences, differences_t)
    # calculate sum of squares between
    SSQ_between = np.sum(sq_differences * reps, axis=0)
    T0 = (SSQ_between/(n_sources -1) - MSwithin)/kappa
    return T0
