import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import logsumexp


'''Functions for model fit'''

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

def TLM_calc_h_sq(X, y):
    """
    X np.array of measurements, rows are objects, columns are variables
    y np 1d-array of labels. labels from {1, ..., n} with n the number of objects. Repetitions get the same label.
    returns: h^2, squared kernel bandwidth
    """

    # get parameters
    m = len(np.unique(y))
    p = X.shape[1]
    # calculate h and h_square
    h = (4/((p+2)*m))**(1/(p+4))
    h_sq = h**2
    return h_sq

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

    # calculate kappa
    # get the number of repetitions per source
    reps = np.array(grouped.size()).reshape((-1, 1))
    # calculate the sum of the repetitions squared and kappa;
    #   kappa represents the average number of repetitions per source
    sum_reps_sq = sum(reps**2)
    n_sources = len(reps)
    kappa = float((reps.sum() - sum_reps_sq/reps.sum())/(n_sources-1))

    # calculate sum_of_squares between
    # substitute rows with their corresponding group means
    group_means = grouped.transform('mean')
    # calculate covariance
    cov_between = group_means.cov(ddof=0)
    # get Sum of Squares Between
    SSQ_between = cov_between * len(group_means)

    # calculate T0: prior between variance
    T0 = (SSQ_between/(n_sources -1) - MSwithin)/kappa
    T0 = T0.to_numpy()
    return T0


'''Functions for predict'''
def TLM_calc_U(X_tr, X_ref, MSwithin, h_sq, T0):
    """
    X_tr np.array of measurements of trace object, rows are repetitions, columns are variables
    X_ref no.array of measurments of reference object, rows are repetitions, columns are variables
    MSwithin np.array: mean within covariance matrix, as calculated by TLM_clc_MSwithin
    h_sq float: square of kernel bandwith
    T0 np.array: between covariance matrix as calculated by TLM_calc_t0
    returns: U_h0, U_hx and U_hn, covariance matrices needed for LR calculation, one for trace, one for ref and
        one for bayesian update of reference means given KDE background means
    """
    # calculate number of trace and reference measurements
    n_trace = len(X_tr)
    n_reference = len(X_ref)
    # Calculate covariance matrices U_h0 and U_hx
    U_h0 = h_sq * T0 + MSwithin/n_trace
    # take the inverse
    U_h0_inv = np.linalg.inv(U_h0)
    U_hx = h_sq * T0 + MSwithin/n_reference
    # Take the inverse
    U_hx_inv = np.linalg.inv(U_hx)
    # Calculate T_hn
    T_hn = h_sq * T0 - np.matmul(np.matmul((h_sq * T0), U_hx_inv), (h_sq * T0))
    # Calculate U_hn
    U_hn = T_hn + MSwithin / n_trace
    # take the inverse
    U_hn_inv = np.linalg.inv(U_hn)
    return U_h0_inv, U_hx_inv, U_hn_inv, U_h0, U_hn

def TLM_calc_mu_h(X_ref, MSwithin, T0, h_sq, X, y):
    """
        X_ref np.array of measurements of reference object, rows are repetitions, columns are variables
        MSwithin = mean within covariance matrix, as calculated by TLM_clc_MSwithin
        h_sq = square of kernel bandwidth
        T0 = between covariance matrix as calculated by TLM_calc_t0
        X = measurements of background data
        returns: mu_h, bayesian update of reference mean given KDE background means
        """
    # calculate number of reference measurements
    n_reference = len(X_ref)
    # calculate mean of reference measurements
    mean_X_reference = np.mean(X_ref, axis=0)
    # calculate precision term, see Bolck et al
    between_precision_X_ref_mean = np.linalg.inv(h_sq * T0 + MSwithin/n_reference)
    # calculate means of background data
    means_z = TLM_calc_means(X, y)
    # calculate the two terms for mu_h and add
    mu_h_1 = np.matmul(np.matmul(h_sq * T0, between_precision_X_ref_mean), mean_X_reference).reshape(-1,1)
    mu_h_2 = np.matmul(np.matmul(MSwithin/n_reference, between_precision_X_ref_mean), means_z.transpose())
    mu_h = mu_h_1 + mu_h_2
    return mu_h.transpose()

def TLM_calc_ln_num(X_trace, X_ref, U_hx_inv, U_hn_inv, mu_h, X, y):
    """
        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        X_ref np.array of measurements of reference object, rows are repetitions, columns are variables
        U_hx_inv, U_hn_inv, np.arrays as calculated by TLM_calc_U
        mu_h np.array with same dimensions as X, calculated by TLM_calc_mu_h
        X: measurements of background data
        y: labels of background data
        returns: ln_num1, natural log of numerator of the LR-formula in Bolck et al.
    """
    # calculate mean of reference and trace measurements
    mean_X_trace = np.mean(X_trace, axis=0).reshape(1,-1)
    mean_X_reference = np.mean(X_ref, axis=0).reshape(1, -1)
    # calculate means of background data
    means_z = TLM_calc_means(X, y)
    # calculate difference matrices
    dif_trace = mean_X_trace - mu_h
    dif_ref = mean_X_reference - means_z
    # calculate matrix products and sums
    ln_num_terms = -0.5 * np.sum(np.matmul(dif_trace, U_hn_inv) * dif_trace, axis=1) + \
        -0.5 * np.sum(np.matmul(dif_ref, U_hx_inv) * dif_ref, axis=1)
    # exponentiate, sum and take log again
    ln_num = logsumexp(ln_num_terms)
    return ln_num

def TLM_calc_ln_den_term(X_ref_or_trace, U_inv, X, y):
    """
        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        X_ref np.array of measurements of reference object, rows are repetitions, columns are variables
        U_hx_inv, U_hn_inv, np.arrays as calculated by TLM_calc_U
        mu_h np.array with same dimensions as X, calculated by TLM_calc_mu_h
        X: measurements of background data
        y: labels of background data
        returns: ln_den_left, natural log of left denominator term of the LR-formula in Bolck et al.
    """
    # calculate mean of reference and trace measurements
    mean_X_ref_or_trace = np.mean(X_ref_or_trace, axis=0).reshape(1, -1)
    # calculate means of background data
    means_z = TLM_calc_means(X, y)
    # calculate difference matrices
    dif_ref = mean_X_ref_or_trace - means_z
    # calculate matrix products and sums
    ln_den_terms = -0.5 * np.sum(np.matmul(dif_ref, U_inv) * dif_ref, axis=1)
    # exponentiate, sum and take log again
    ln_den_term = logsumexp(ln_den_terms)
    return ln_den_term


def TLM_calc_log10_LR(U_h0, U_hn, ln_num, ln_den_left, ln_den_right, y):
    """
        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        U_h0, U_hn, np.arrays as calculated by TLM_calc_U
        ln_num, ln_den_left, ln_den_right: terms in big fraction in Bolck et al
        X: measurements of background data
        y: labels of background data
        returns: log10_LR, 10log of LR according to the LR-formula in Bolck et al.
    """
    # get number of sources in y
    m = len(np.unique(y))
    # calculate ln LR and change base to 10log
    ln_LR = np.log(m) - 0.5 * np.log(np.linalg.det(U_hn)) + 0.5 * np.log(np.linalg.det(U_h0)) + ln_num - ln_den_left - ln_den_right
    log10_LR = ln_LR/np.log(10)
    return log10_LR

def TLM_predict_log10_LR(X_trace, X_ref, MSwithin, h_sq, T0, X, y):
    """
        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        X_ref np.array of measurements of reference object, rows are repetitions, columns are variables
        MSwithin, h_sq, T0: calculated through TLM_fit functions
        X: measurements of background data
        y: labels of background data
        returns: ln_den_right, natural log of right denominator term of the LR-formula in Bolck et al.
    """
    # do intermediate calculations
    U_h0_inv, U_hx_inv, U_hn_inv, U_h0, U_hn = TLM_calc_U(X_trace, X_ref, MSwithin, h_sq, T0)
    mu_h = TLM_calc_mu_h(X_ref, MSwithin, T0, h_sq, X, y)
    ln_num = TLM_calc_ln_num(X_trace, X_ref, U_hx_inv, U_hn_inv, mu_h, X, y)
    ln_den_left = TLM_calc_ln_den_term(X_trace, U_h0_inv, X, y)
    ln_den_right = TLM_calc_ln_den_term(X_ref, U_h0_inv, X, y)
    # calculate log10_LR
    log10_LR = TLM_calc_log10_LR(U_h0, U_hn, ln_num, ln_den_left, ln_den_right, y)
    return log10_LR
