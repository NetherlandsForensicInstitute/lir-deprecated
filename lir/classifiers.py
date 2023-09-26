import pandas as pd
import numpy as np

def TLM_calc_mean_covariance_within(X, y):
    """
    X np.array of measurements, rows are sources/repetitions, columns are features
    y np 1d-array of labels. labels from {1, ..., n} with n the number of sources. Repetitions get the same label.
    returns: mean within covariance matrix, np.array

    This function calculates a matrix of mean covariances within each of the sources, it does so by grouping the data
    per source, calculating the covariance matrices per source and then taking the mean per feature.
    """
    # use pandas functionality to allow easy calculation
    df = pd.DataFrame(X, index=pd.Index(y, name="label"))
    # filter out single-repetitions,since they do not contribute to covariance calculations
    grouped = df.groupby(by='label')
    filtered = grouped.filter(lambda x: x[0].count() > 1)
    # make groups again by source id and calculate covariance matrices per source
    grouped = filtered.groupby(by='label')
    covars = grouped.cov(ddof=1)
    # add index names to allow grouping by feature, group by feature and get mean covariance matrix
    covars.index.names = ["Source", "Feature"]
    grouped_by_feature = covars.groupby(["Feature"])
    mean_covars = np.array(grouped_by_feature.mean())
    return mean_covars

