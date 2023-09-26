import pandas as pd
import numpy as np


class TwoLevelModel:
    def __init__(self):
        """
        An implementation of the two-level model as outlined in FSI191(2009)42 by Bolck et al. "Different likelihood ratio approaches to evaluate the strength of evidence of
            MDMA tablet comparisons".
        As this model uses a different dataset for training than the dataset that is to be used by the calibrator,
        it is not fit for use with lir.CalibratedScorer().


        Model description:

        Definitions
        X_ij = vector, measurement of reference j, ith repetition
        Y_kl = vector, measurement of trace l, kth repetition
        The number of repetitions for X = n and for Y = m

        Model:

        First level of variance:
        X_ij ~ N(theta_j, sigma_within)
        Y_kl ~ N(theta_k, sigma_within)
        , where theta_j is the true (but unknown) mean of the reference and theta_k the true but unknown mean of the trace.
        sigma_within is assumed equal for trace and reference (and for repeated measurements of some background data)

        Second level of variance:
        theta_j ~ theta_k ~ KDE(means background database, h)
        with h the kernel bandwidth.

        H1: theta_j = theta_k
        H2: theta_j independent of theta_k

        Numerator LR = Integral_theta N(X_Mean|theta, sigma_within, n) * N(Y_mean|theta, sigma_within, m) * KDE(theta|means background database, h)
        Denominator LR = Integral_theta N(X_Mean|theta, sigma_within, n) * KDE(theta|means background database, h) * Integral_theta N(Y_Mean|theta, sigma_within, m) * KDE(theta|means background database, h)

        In Bolck et al in the appendix one finds a closed-form solution for the evaluation of these integrals.

        sigma_within and h (and other parameters) are estimated from repeated measurements of background data.
        """
        self.model_fitted = False
        self.X = None
        self.y = None
        self.mean_covars = None
        self.means_per_source = None
        self.kernel_bandwidth_sq = None
        self.between_covars = None

    def fit(self, X, y):
        """
        X np.array of measurements, rows are sources/repetitions, columns are features
        y np 1d-array of labels. For each source a unique identifier (label). Repetitions get the same label.

        Construct the necessary matrices/scores/etc based on test data (X) so that we can predict a score later on.
        Store any calculated parameters in `self`.
        """
        self.model_fitted = True
        self.X = X
        self.y = y
        self.mean_covars = self.fit_mean_covariance_within(self.X, self.y)
        self.means_per_source = self.fit_means_per_source(self.X, self.y)
        self.kernel_bandwidth_sq = self.fit_kernel_bandwidth_squared(self.X, self.y)
        self.between_covars = self.fit_between_covariance(self.X, self.y)


    def transform(self, X):
        """
        Predict odds scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).
        """
        if self.model_fitted:
            raise ValueError("The model is not fitted; fit it before you use it for predicting")
        return self.model_fitted

    def predict_proba(self, X):
        """
        Predict probability scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).
        """
        if self.model_fitted:
            raise ValueError("The model is not fitted; fit it before you use it for predicting")
        return self.model_fitted

    def _fit_mean_covariance_within(self, X, y):
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

    def _fit_means_per_source(self, X, y):
        """
        X np.array of measurements, rows are sources/repetitions, columns are features
        y np 1d-array of labels. For each source a unique identifier (label). Repetitions get the same label.
        returns: means per source in a np.array matrix of size: number of sources * number of features
        """
        # use pandas functionality to allow easy calculation
        # Group by source
        df = pd.DataFrame(X, index=pd.Index(y, name="label"))
        grouped = df.groupby(by='label')
        # Calculate mean per source and convert to numpy array
        means = np.array(grouped.mean())
        return means

    def _fit_kernel_bandwidth_squared(self, X, y):
        """
        X np.array of measurements, rows are sources/repetitions, columns are features
        y np 1d-array of labels. For each source a unique identifier (label). Repetitions get the same label.
        returns: squared kernel bandwidth for the kernel density estimator with a normal kernel
            (using Silverman's rule for multivariate data)

        Reference: 'Density estimation for statistics and data analysis', B.W. Silverman,
            page 86 formula 4.14 with A(K) the second row in the table on page 87
        """
        # get number of sources and number of features
        n_sources = len(np.unique(y))
        n_features = X.shape[1]
        # calculate kernel bandwidth and square it, using Silverman's rule for multivariate data
        kernel_bandwidth = (4 / ((n_features + 2) * n_sources)) ** (1 / (n_features + 4))
        kernel_bandwidth_sq = kernel_bandwidth ** 2
        return kernel_bandwidth_sq

    def _fit_between_covariance(self, X, y):

        """
        X np.array of measurements, rows are objects, columns are variables
        y np 1d-array of labels. labels from {1, ..., n} with n the number of objects. Repetitions get the same label.
        returns: estimated covariance of true mean of the features between sources in the population in a np.array
            square matrix with number of features^2 as dimension
        """
        # this function should not be called on its own, but only within 'fit'
        if self.model_fitted == False:
            raise ValueError("This function should only be used within 'fit'")
        else:
            # use pandas functionality to allow easy calculation
            df = pd.DataFrame(X, index=pd.Index(y, name="label"))
            # group per source
            grouped = df.groupby(by='label')

            # calculate kappa; kappa represents the "average" number of repetitions per source
            # get the repetitions per source
            reps = np.array(grouped.size()).reshape((-1, 1))
            # calculate the sum of the repetitions squared and kappa
            sum_reps_sq = sum(reps ** 2)
            n_sources = len(reps)
            kappa = float((reps.sum() - sum_reps_sq / reps.sum()) / (n_sources - 1))

            # calculate sum_of_squares between
            # substitute rows with their corresponding group means
            group_means = grouped.transform('mean')
            # calculate covariance of measurements
            cov_between_measurement = group_means.cov(ddof=0)
            # get Sum of Squares Between
            SSQ_between = cov_between_measurement * len(group_means)

            # calculate between covariance matrix
            # Kappa converts within variance at measurement level to within variance at mean of source level and
            #   scales the SSQ_between to a mean between variance
            between_covars = (SSQ_between / (n_sources - 1) - self.mean_covars) / kappa
            between_covars = between_covars.to_numpy()
            return between_covars