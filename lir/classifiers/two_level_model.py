import pandas as pd
import numpy as np
from scipy.special import logsumexp


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
        self.n_features_train = None
        self.n_sources = None
        self.mean_within_covars = None
        self.means_per_source = None
        self.kernel_bandwidth_sq = None
        self.between_covars = None
        self.covars_trace = None
        self.covars_ref = None
        self.covars_trace_update = None
        self.covars_trace_inv = None
        self.covars_ref_inv = None
        self.covars_trace_update_inv = None
        self.updated_ref_mean = None

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
        self.mean_within_covars = self._fit_mean_covariance_within(self.X, self.y)
        self.means_per_source = self._fit_means_per_source(self.X, self.y)
        self.kernel_bandwidth_sq = self._fit_kernel_bandwidth_squared(self.X, self.y)
        self.between_covars = self._fit_between_covariance(self.X, self.y)
        self.n_sources = self._fit_n_sources(self.y)
        self.n_features_train = self._get_n_features(self.X)


    def transform(self, X_trace, X_ref):
        """
        Predict odds scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).
        """
        log10_LR_score = self._predict_log10_LR_score(X_trace, X_ref)
        odds_score = 10**log10_LR_score
        return odds_score

    def predict_proba(self, X_trace, X_ref):
        """
        Predict probability scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).
        """
        odds_score = self.transform(X_trace, X_ref)
        p0  = odds_score/(1-odds_score)
        p1 = 1 - p0
        return np.array([p0, p1])

    def _predict_log10_LR_score(self, X_trace, X_ref):
        """
        Predict ln_LR scores, making use of the parameters constructed during `self.fit()` (which should
                now be stored in `self`).

        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        X_ref np.array of measurements of reference object, rows are repetitions, columns are variables
        returns: log10_LR_score, log10 LR according to the two_level_model in Bolck et al.
        """
        if self.model_fitted == False:
            raise ValueError("The model is not fitted; fit it before you use it for predicting")
        elif self._get_n_features(X_trace) != self.n_features_train:
            raise ValueError("The numberof features in the training data is different from the number of features in the trace")
        elif self._get_n_features(X_ref) != self.n_features_train:
            raise ValueError("The number of features in the training data is different from the number of features in the reference")
        else:
            covars_trace, covars_trace_update, covars_ref, covars_trace_inv, covars_trace_update_inv, covars_ref_inv = self._predict_covariances_trace_ref(X_trace, X_ref)
            updated_ref_mean = self._predict_updated_ref_mean(X_ref, covars_ref_inv)
            ln_num = self._predict_ln_num(X_trace, X_ref, covars_ref_inv, covars_trace_update_inv, updated_ref_mean)
            ln_den_left = self._predict_ln_den_term(X_ref, covars_ref_inv)
            ln_den_right = self._predict_ln_den_term(X_trace, covars_trace_inv)
            # calculate log10_LR
            log10_LR_score = self._predict_log10_LR_from_formula_Bolck(covars_trace, covars_trace_update, ln_num, ln_den_left, ln_den_right)
            return log10_LR_score

    def _get_n_features(self, X):
        n_features = X.shape[1]
        return n_features

    def _fit_n_sources(self, y):
        """
        y np 1d-array of labels. labels from {1, ..., n} with n the number of sources. Repetitions get the same label.
        returns: number of sources in y (int)
        """
        # get number of sources in y
        n_sources = len(np.unique(y))
        return n_sources

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
        # calculate kernel bandwidth and square it, using Silverman's rule for multivariate data
        kernel_bandwidth = (4 / ((self.n_features_train + 2) * self.n_sources)) ** (1 / (self.n_features_train + 4))
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
            kappa = ((reps.sum() - sum_reps_sq / reps.sum()) / (n_sources - 1)).item()

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
            between_covars = (SSQ_between / (n_sources - 1) - self.mean_within_covars) / kappa
            between_covars = between_covars.to_numpy()
            return between_covars

    def _predict_covariances_trace_ref(self, X_trace, X_ref):
        """
        X_tr np.array of measurements of trace object, rows are repetitions, columns are features
        X_ref np.array of measurements of reference object, rows are repetitions, columns features
        returns: covariance matrices of the trace and reference data and their respective inverses needed for
        LR calculation;
            covars_trace is the covariance matrix for the trace data given a KDE background mean (U_h0),
            covars_trace_update is the covariance matrix for the trace mean with a bayesian update of reference mean
            given a KDE background mean (U_hn),
            covars_ref is the covariance matrix for the reference data given a KDE background mean (U_hx),
            covars_trace_inv is the inverse of covars_trace,
            covars_trace_update_inv is the inverse of covars_trace_update,
            covars_ref_inv is the inverse of covars_ref
        """
        # this function should not be called on its own,
        if self.model_fitted == False:
            raise ValueError("This function should only be used within 'fit'")

        # Number of trace and reference measurements
        n_trace = len(X_trace)
        n_reference = len(X_ref)
        # Calculate covariance matrix for the trace data, given the training data (U_h0)
        covars_trace = self.kernel_bandwidth_sq * self.between_covars + self.mean_within_covars / n_trace
        # Calculate covariance matrix for the reference data, given the training data (U_hx)
        covars_ref = self.kernel_bandwidth_sq * self.between_covars + self.mean_within_covars / n_reference
        # take the inverses
        covars_trace_inv = np.linalg.inv(covars_trace)
        covars_ref_inv = np.linalg.inv(covars_ref)
        # Calculate T_hn
        T_hn = self.kernel_bandwidth_sq * self.between_covars - \
               np.matmul(np.matmul((self.kernel_bandwidth_sq * self.between_covars), covars_ref_inv),
                         (self.kernel_bandwidth_sq * self.between_covars))
        # Calculate covariance matrix for the trace data, given the training data and with a Bayesian update with
        #   the reference data under Hp (U_hn)
        covars_trace_update = T_hn + self.mean_within_covars / n_trace
        # take the inverse
        covars_trace_update_inv = np.linalg.inv(covars_trace_update)
        # question: covars_trace redundant to return?
        return covars_trace, covars_trace_update, covars_ref, covars_trace_inv, covars_trace_update_inv, covars_ref_inv

    def _predict_updated_ref_mean(self, X_ref, covars_ref_inv):
        """
        X_ref np.array of measurements of reference object, rows are repetitions, columns features
        returns: updated_ref_mean, bayesian update of reference mean given KDE background means
        """
        # calculate number of reference measurements
        n_reference = len(X_ref)
        # calculate mean of reference measurements
        mean_X_reference = np.mean(X_ref, axis=0)
        # calculate the two terms for mu_h and add, see Bolck et al
        mu_h_1 = np.matmul(np.matmul(self.kernel_bandwidth_sq * self.between_covars, covars_ref_inv), mean_X_reference).reshape(-1, 1)
        mu_h_2 = np.matmul(np.matmul(self.mean_within_covars / n_reference, covars_ref_inv), self.means_per_source.transpose())
        updated_ref_mean_T = mu_h_1 + mu_h_2
        updated_ref_mean = updated_ref_mean_T.transpose()
        return updated_ref_mean

    def _predict_ln_num(self, X_trace, X_ref, covars_ref_inv, covars_trace_update_inv, updated_ref_mean):
        """
        See Bolck et al formula in appendix. The formula consists of three sum_terms (and some other terms). The numerator sum term is calculated here.
        The numerator is based on the product of two Gaussion PDFs.
        The first PDF: ref_mean ~ N(background_mean, U_hx).
        The second PDF: trace_mean ~ N(updated_ref_mean, U_hn).
        In this function log of the PDF is taken (so the exponentiation is left out and the product becomes a sum).

        X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
        X_ref np.array of measurements of reference object, rows are repetitions, columns are variables
        covars_ref_inv, covars_trace_update_inv, np.arrays as calculated by _predict_covariances_trace_ref
        updated_ref_mean np.array with same dimensions as X, calculated by _predict_updated_ref_mean
        returns: ln_num1, natural log of numerator of the LR-formula in Bolck et al.
        """
        # calculate mean of reference and trace measurements
        mean_X_trace = np.mean(X_trace, axis=0).reshape(1, -1)
        mean_X_reference = np.mean(X_ref, axis=0).reshape(1, -1)
        # calculate difference vectors (in matrix form)
        dif_trace = mean_X_trace - updated_ref_mean
        dif_ref = mean_X_reference - self.means_per_source
        # calculate matrix products and sums
        ln_num_terms = -0.5 * np.sum(np.matmul(dif_trace, covars_trace_update_inv) * dif_trace, axis=1) + \
                       -0.5 * np.sum(np.matmul(dif_ref, covars_ref_inv) * dif_ref, axis=1)
        # exponentiate, sum and take log again
        ln_num = logsumexp(ln_num_terms)
        return ln_num

    def _predict_ln_den_term(self, X_ref_or_trace, covars_inv):
        """
        See Bolck et al formula in appendix. The formula consists of three sum_terms (and some other terms). A denominator sum term is calculated here.

        X_ref_or_trace np.array of measurements of reference or trace object, rows are repetitions, columns are features
        U_inv, np.array with respective covariance matrix as calculated by _predict_covariances_trace_ref
        returns: ln_den, natural log of a denominator term of the LR-formula in Bolck et al.
        """
        # calculate mean of reference or trace measurements
        mean_X_ref_or_trace = np.mean(X_ref_or_trace, axis=0).reshape(1, -1)
        # calculate difference vectors (in matrix form)
        dif_ref = mean_X_ref_or_trace - self.means_per_source
        # calculate matrix products and sums
        ln_den_terms = -0.5 * np.sum(np.matmul(dif_ref, covars_inv) * dif_ref, axis=1)
        # exponentiate, sum and take log again
        ln_den_term = logsumexp(ln_den_terms)
        return ln_den_term

    def _predict_log10_LR_from_formula_Bolck(self, covars_trace, covars_trace_update, ln_num, ln_den_left, ln_den_right):
        """
            X_trace np.array of measurements of trace object, rows are repetitions, columns are variables
            covars_trace, covars_trace_update, np.arrays as calculated by _predict_covariances_trace_ref
            ln_num, ln_den_left, ln_den_right: terms in big fraction in Bolck et al, as calculated by _predict_ln_num
                and _predict_ln_den_term
            returns: log10_LR_score, 10log of LR according to the LR-formula in Bolck et al.
        """
        # calculate ln LR_score and change base to 10log
        ln_LR_score = np.log(self.n_sources) - 0.5 * np.log(np.linalg.det(covars_trace_update)) + \
                      0.5 * np.log(np.linalg.det(covars_trace)) + ln_num - ln_den_left - ln_den_right
        log10_LR_score = ln_LR_score / np.log(10)
        return log10_LR_score


