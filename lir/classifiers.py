import pandas as pd
import numpy as np

def TLM_calc_MSwithin(X, y):
    """
    X np.array of measurements, rows are sources/repetitions, columns are features
    y np 1d-array of labels. labels from {1, ..., n} with n the number of sources. Repetitions get the same label.
    returns: mean within covariance matrix, np.array
    """
    # use pandas functionality to allow easy calculation
    df = pd.DataFrame(X, index=pd.Index(y, name="label"))
    # filter out single-repetitions,since they do not contribute to covariance calculations
    grouped = df.groupby(by='label')
    filtered = grouped.filter(lambda x: x[0].count() > 1)
    # make groups again by windownr and calculate covariance matrices per window
    grouped = filtered.groupby(by='label')
    covars = grouped.cov(ddof=1)
    # add index names to allow grouping by element, group by element and get mean covariance matrix
    covars.index.names = ["Source", "Feature"]
    grouped_by_element = covars.groupby(["Feature"])
    mean_covars = grouped_by_element.mean()
    return np.array(mean_covars)

