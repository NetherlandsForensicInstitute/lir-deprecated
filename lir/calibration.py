import logging
import math
import numpy as np
import warnings
from functools import partial
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from typing import Optional, Tuple, Union, Iterable, Callable, Sized

from .bayeserror import elub
from .loss_functions import negative_log_likelihood_balanced
from .regression import IsotonicRegressionInf
from .util import Xy_to_Xn, to_odds, ln_to_log10, Bind

LOG = logging.getLogger(__name__)


def check_misleading_Inf_negInf(log_odds_X, y):
    """
    for calibration training on log_odds domain. Check whether negInf under H1 and Inf under H2 occurs and give error if so.
    """

    # give error message if H1's contain zeros and H2's contain ones
    if np.any(np.isneginf(log_odds_X[y == 1])) and np.any(np.isposinf(log_odds_X[y == 0])):
        raise ValueError('Your data is possibly problematic for this calibrator. You have negInf under H1 and Inf under H2 after logodds transform. If you really want to proceed, adjust probs in order to get finite values on the logodds domain')
    # give error message if H1's contain zeros
    if np.any(np.isneginf(log_odds_X[y == 1])):
        raise ValueError('Your data is possibly problematic for this calibrator. You have negInf under H1 after logodds transform. If you really want to proceed, adjust probs in order to get finite values on the logodds domain')
    # give error message if H2's contain ones
    if np.any(np.isposinf(log_odds_X[y == 0])):
        raise ValueError('Your data is possibly problematic for this calibrator. You have Inf under H2 after logodds transform. If you really want to proceed, adjust probs in order to get finite values on the logodds domain')


def compensate_and_remove_negInf_Inf(log_odds_X, y):
    """
    for Gaussian and KDE-calibrator fitting: remove negInf, Inf and compensate
    """
    X_finite = np.isfinite(log_odds_X)
    el_H1 = np.logical_and(X_finite, y == 1)
    el_H2 = np.logical_and(X_finite, y == 0)
    n_H1 = np.sum(y)
    numerator = np.sum(el_H1)/n_H1
    denominator = np.sum(el_H2)/(len(y)-n_H1)
    y = y[X_finite]
    log_odds_X = log_odds_X[X_finite]
    return log_odds_X, y, numerator, denominator


class NormalizedCalibrator(BaseEstimator, TransformerMixin):
    """
    Normalizer for any calibration function.

    Scales the probability density function of a calibrator so that the
    probability mass is 1.
    """

    def __init__(self, calibrator, add_one=False, sample_size=100, value_range=(0, 1)):
        self.calibrator = calibrator
        self.add_one = add_one
        self.value_range = value_range
        self.step_size = (value_range[1] - value_range[0]) / sample_size

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        self.X0n = X0.shape[0]
        self.X1n = X1.shape[0]
        self.calibrator.fit(X, y)
        self.calibrator.transform(np.arange(self.value_range[0], self.value_range[1], self.step_size))
        self.p0mass = np.sum(self.calibrator.p0) / 100
        self.p1mass = np.sum(self.calibrator.p1) / 100
        return self

    def transform(self, X):
        self.calibrator.transform(X)
        self.p0 = self.calibrator.p0 / self.p0mass
        self.p1 = self.calibrator.p1 / self.p1mass
        if self.add_one:
            self.p0 = self.X0n / (self.X0n + 1) * self.p0 + 1 / self.X0n
            self.p1 = self.X1n / (self.X1n + 1) * self.p1 + 1 / self.X1n
        return self.p1 / self.p0

    def __getattr__(self, name):
        return getattr(self.calibrator, name)


class ScalingCalibrator(BaseEstimator, TransformerMixin):
    """
    Calibrator which adjusts the LRs towards 1 depending on the sample size.

    This is done by adding a value of 1/sample_size to the probabilities of the underlying calibrator and
    scaling the result.
    """

    def __init__(self, calibrator):
        self.calibrator = calibrator

    def fit(self, X, y):
        self.calibrator.fit(X, y)
        X0, X1 = Xy_to_Xn(X, y)
        self.X0n = X0.shape[0]
        self.X1n = X1.shape[0]
        return self

    def transform(self, X):
        self.calibrator.transform(X)
        self.p0 = self.X0n / (self.X0n + 1) * self.calibrator.p0 + 1 / self.X0n
        self.p1 = self.X1n / (self.X1n + 1) * self.calibrator.p1 + 1 / self.X1n
        return self.p1 / self.p0

    def __getattr__(self, name):
        return getattr(self.calibrator, name)


class FractionCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of the distance of a score value to the
    extremes of its value range.
    """

    def __init__(self, value_range=(0, 1)):
        self.value_range = value_range

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        self._abs_points0 = np.abs(self.value_range[0] - X0)
        self._abs_points1 = np.abs(self.value_range[1] - X1)
        return self

    def density(self, X, class_value, points):
        X = np.abs(self.value_range[class_value] - X)

        numerator = np.array([points[points >= x].shape[0] for x in X])
        denominator = len(points)
        return numerator / denominator

    def transform(self, X):
        X = np.array(X)
        self.p0 = self.density(X, 0, self._abs_points0)
        self.p1 = self.density(X, 1, self._abs_points1)

        with np.errstate(divide='ignore'):
            return self.p1 / self.p0


class KDECalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, bandwidth: Union[Callable, str, float, Tuple[float, float]] = None):
        """

        :param bandwidth:
            * If bandwidth has a float value, this value is used as the bandwidth for both distributions.
            * If bandwidth is a tuple, it should contain two floating point values: the bandwidth for the distribution
              of the classes with labels 0 and 1, respectively.
            * If bandwidth has the str value "silverman", Silverman's rule of thumb is used as the bandwidth for both
              distributions separately.
            * If bandwidth is callable, it should accept two arguments, `X` and `y`, and return a tuple of two values
              which are the bandwidths for the two distributions.
        """
        if bandwidth is None:
            warnings.warn("missing bandwidth argument for KDE, defaulting to silverman (default argument will be removed in the future)")
            bandwidth = "silverman"
        self.bandwidth: Callable = self._parse_bandwidth(bandwidth)
        self._kde0: Optional[KernelDensity] = None
        self._kde1: Optional[KernelDensity] = None
        self.numerator, self.denominator = None, None

    @staticmethod
    def bandwidth_silverman(X, y):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0

        bandwidth = []
        for label in np.unique(y):
            values = X[y == label]
            std = np.std(values)
            if std == 0:
                # can happen eg if std(values) = 0
                warnings.warn('silverman bandwidth cannot be calculated if standard deviation is 0', RuntimeWarning)
                LOG.info('found a silverman bandwidth of 0 (using dummy value)')
                std = 1

            v = math.pow(std, 5) / len(values) * 4. / 3
            bandwidth.append(math.pow(v, .2))

        return bandwidth

    @staticmethod
    def bandwidth_scott(X, y):
        """
        Not implemented.
        """
        raise

    def fit(self, X, y):
        # check if data is sane
        check_misleading_Inf_negInf(X, y)

        # KDE needs finite scale. Inf and negInf are treated as point masses at the extremes.
        # Remove them from data for KDE and calculate fraction data that is left.
        # LRs in finite range will be corrected for fractions in transform function
        X, y, self.numerator, self.denominator = compensate_and_remove_negInf_Inf(X, y)
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)

        bandwidth0, bandwidth1 = self.bandwidth(X, y)
        self._kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(X0)
        self._kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X1)
        return self

    def transform(self, X):
        assert self._kde0 is not None, "KDECalibrator.transform() called before fit"
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))
        # initiate LRs_output
        LLRs_output = np.empty(np.shape(X))

        # get inf and neginf
        wh_inf = np.isposinf(X)
        wh_neginf = np.isneginf(X)

        # assign hard values for extremes
        LLRs_output[wh_inf] = np.inf
        LLRs_output[wh_neginf] = -np.inf
        self.p0[wh_inf] = 0
        self.p1[wh_inf] = 1
        self.p0[wh_neginf] = 1
        self.p1[wh_neginf] = 0

        # get elements that are not inf or neginf
        el = np.isfinite(X)
        X = X[el]

        # perform KDE as usual
        X = X.reshape(-1, 1)
        ln_H1 = self._kde1.score_samples(X)
        ln_H2 = self._kde0.score_samples(X)
        ln_dif = ln_H1 - ln_H2
        log10_dif = ln_to_log10(ln_dif)

        #calculate p0 and p1's (redundant?)
        self.p0[el] = self.denominator * np.exp(ln_H2)
        self.p1[el] = self.numerator * np.exp(ln_H1)

        # apply correction for fraction of negInf and Inf data
        log10_compensator = np.log10(self.numerator / self.denominator)
        LLRs_output[el] = log10_compensator + log10_dif
        return np.float_power(10, LLRs_output)

    @staticmethod
    def _parse_bandwidth(bandwidth: Union[Callable, float, Tuple[float, float]]) \
            -> Callable:
        """
        Returns bandwidth as a tuple of two (optional) floats.
        Extrapolates a single bandwidth
        :param bandwidth: provided bandwidth
        :return: bandwidth used for kde0, bandwidth used for kde1
        """
        assert bandwidth is not None, "KDE requires a bandwidth argument"
        if callable(bandwidth):
            return bandwidth
        elif bandwidth == "silverman":
            return KDECalibrator.bandwidth_silverman
        elif bandwidth == "scott":
            return KDECalibrator.bandwidth_scott
        elif isinstance(bandwidth, str):
            raise ValueError(f"invalid input for bandwidth: {bandwidth}")
        elif isinstance(bandwidth, Sized):
            assert len(bandwidth) == 2, f"bandwidth should have two elements; found {len(bandwidth)}; bandwidth = {bandwidth}"
            return lambda X, y: bandwidth
        else:
            return lambda X, y: (0+bandwidth, bandwidth)


class KDECalibratorInProbabilityDomain(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, bandwidth: Union[Callable, str, float, Tuple[float, float]] = None):
        """

        :param bandwidth:
            * If None is provided the Silverman's rule of thumb is
            used to calculate the bandwidth for both distributions (independently)
            * If a single float is provided this is used as the bandwith for both
            distributions
            * If a tuple is provided, the first entry is used for the bandwidth
            of the first distribution (kde0) and the second entry for the second
            distribution (if value is None: Silverman's rule of thumb is used)
        """

        warnings.warn(f"the class {type(self).__name__} will be removed in the future")
        if bandwidth is None:
            warnings.warn("missing bandwidth argument for KDE, defaulting to 1 (default argument will be removed in the future)")
            bandwidth = (1, 1)
        self.bandwidth: Callable = KDECalibrator._parse_bandwidth(bandwidth)
        self._kde0: Optional[KernelDensity] = None
        self._kde1: Optional[KernelDensity] = None

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)

        bandwidth0, bandwidth1 = self.bandwidth(X, y)
        self._kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(X0)
        self._kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X1)
        return self

    def transform(self, X):
        assert self._kde0 is not None, "KDECalibrator.transform() called before fit"

        X = X.reshape(-1, 1)
        self.p0 = np.exp(self._kde0.score_samples(X))
        self.p1 = np.exp(self._kde1.score_samples(X))

        with np.errstate(divide='ignore'):
            return self.p1 / self.p0


class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.
    """
    
    def __init__(self, **kwargs):
        self._logit = LogisticRegression(class_weight='balanced', **kwargs)

    def fit(self, X, y):

        # sanity check
        check_misleading_Inf_negInf(X, y)

        # if data is sane, remove Inf under H1 and minInf under H2 from the data if present (if present, these prevent logistic regression to train while the loss is zero, so they can be safely removed)
        el = np.isfinite(X)
        y = y[el]
        X = X[el]

        # train logistic regression
        X = X.reshape(-1, 1)
        self._logit.fit(X, y)
        return self

    def transform(self, X):

        # initiate LLRs_output
        LLRs_output = np.empty(np.shape(X))
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))

        # get boundary log_odds values
        zero_elements = np.where(X == -1 * np.inf)
        ones_elements = np.where(X == np.inf)

        # assign desired output for these boundary values to LLRs_output
        LLRs_output[zero_elements] = np.multiply(-1, np.inf)
        LLRs_output[ones_elements] = np.inf

        # get elements with values between negInf and Inf (the boundary values)
        between_elements = np.all(np.array([X != np.inf, X != -1 * np.inf]), axis=0)

        # get LLRs for X[between_elements]
        LnLRs = self._logit.intercept_ + self._logit.coef_ * X[between_elements]
        LLRs = ln_to_log10(LnLRs)
        LLRs = np.reshape(LLRs, np.sum(between_elements))
        LLRs_output[between_elements] = LLRs

        # calculation of self.p1 and self.p0 is redundant?
        self.p1[zero_elements] = 0
        self.p1[ones_elements] = 1
        self.p1[between_elements] = self._logit.predict_proba(X[between_elements].reshape(-1, 1))[:, 1]
        self.p0 = 1 - self.p1
        return np.float_power(10, LLRs_output)


class LogitCalibratorInProbabilityDomain(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.
    """
    def __init__(self, **kwargs):
        warnings.warn(f"the class {type(self).__name__} will be removed in the future")
        self._logit = LogisticRegression(class_weight='balanced', **kwargs)

    def fit(self, X, y):
        # train logistic regression
        X = X.reshape(-1, 1)
        self._logit.fit(X, y)
        return self

    def transform(self, X):

        # calculation of self.p1 and self.p0 is redundant?
        self.p1 = self._logit.predict_proba(X.reshape(-1, 1))[:, 1]  # probability of class 1
        self.p0 = (1 - self.p1)

        # get LLRs for X
        LnLRs = np.add(self._logit.intercept_, np.multiply(self._logit.coef_, X))
        LLRs = ln_to_log10(LnLRs)
        LLRs = LLRs.reshape(len(X))
        return np.float_power(10, LLRs)


class GaussianCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses a gaussian mixture model for interpolation.

    First it transforms data (which is in probability domain) to logods domain
    """
    def __init__(self, n_components_H0=1, n_components_H1=1):
        self.n_components_H1 = n_components_H1
        self.n_components_H0 = n_components_H0
        self.numerator = None
        self.denominator = None
        self._model0: Optional[GaussianMixture] = None
        self._model1: Optional[GaussianMixture] = None

    def fit(self, X, y):
        #check whether training data is sane
        check_misleading_Inf_negInf(X, y)

        # Gaussian mixture needs finite scale. Inf and negInf are treated as point masses at the extremes.
        # Remove them from data for Gaussian mixture and calculate fraction data that is left.
        # LRs in finite range will be corrected for fractions in transform function
        X, y, self.numerator, self.denominator = compensate_and_remove_negInf_Inf(X, y)

        # perform Gaussian mixture as usual
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        self._model0 = GaussianMixture(n_components=self.n_components_H0).fit(X0)
        self._model1 = GaussianMixture(n_components=self.n_components_H1).fit(X1)
        return self

    def transform(self, X):
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))

        # initiate LLRs_output
        LLRs_output = np.empty(np.shape(X))

        # get inf and neginf
        wh_inf = np.isposinf(X)
        wh_neginf = np.isneginf(X)

        # assign hard values for extremes
        LLRs_output[wh_inf] = np.inf
        LLRs_output[wh_neginf] = -1 * np.inf
        self.p0[wh_inf] = 0
        self.p1[wh_inf] = 1
        self.p0[wh_neginf] = 1
        self.p1[wh_neginf] = 0

        # get elements that are not inf or neginf
        el = np.isfinite(X)
        X = X[el]

        #perform density calculations for X as usual
        X = X.reshape(-1, 1)
        ln_H1 = self._model1.score_samples(X)
        ln_H2 = self._model0.score_samples(X)
        ln_dif = ln_H1 - ln_H2
        log10_dif = ln_to_log10(ln_dif)

        # calculation of p0 and p1's redundant?
        self.p0[el] = np.multiply(self.denominator, np.exp(ln_H2))
        self.p1[el] = np.multiply(self.numerator, np.exp(ln_H1))

        #apply correction for fraction of Infs and negInfs
        log10_compensator = np.add(np.log10(self.numerator), np.multiply(-1, np.log(self.denominator)))
        LLRs_output[el] = np.add(log10_compensator, log10_dif)
        return np.float_power(10, LLRs_output)


class GaussianCalibratorInProbabilityDomain(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses a gaussian mixture model for interpolation.

    Fits Gaussian on probabilities
    """

    def __init__(self, n_components_H0=1, n_components_H1=1):
        warnings.warn(f"the class {type(self).__name__} will be removed in the future")
        self.n_components_H1 = n_components_H1
        self.n_components_H0 = n_components_H0

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        self._model0 = GaussianMixture(n_components=self.n_components_H0).fit(X0)
        self._model1 = GaussianMixture(n_components=self.n_components_H1).fit(X1)
        return self

    def transform(self, X):
        X = X.reshape(-1, 1)
        self.p0 = np.exp(self._model0.score_samples(X))
        self.p1 = np.exp(self._model1.score_samples(X))
        return self.p1 / self.p0


class IsotonicCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses isotonic regression for interpolation.
    """

    def __init__(self, add_one=False, add_misleading=0):
        """
        Arguments:
            add_one: deprecated (same as add_misleading=1)
            add_misleading: int: add misleading data points on both sides (default: 0)
        """
        if add_one:
            warnings.warn('parameter `add_one` is deprecated; use `add_misleading=1` instead')

        self.add_misleading = (1 if add_one else 0) + add_misleading
        self._ir = IsotonicRegressionInf(out_of_bounds='clip')

    def fit(self, X, y, **fit_params):
        assert np.all(np.unique(y) == np.arange(2)), 'y labels must be 0 and 1'

        # prevent extreme LRs
        if 'add_misleading' in fit_params:
            n_misleading = fit_params['add_misleading']
        elif 'add_one' in fit_params:
            warnings.warn('parameter `add_one` is deprecated; use `add_misleading=1` instead')
            n_misleading = 1 if fit_params['add_one'] else 0
        else:
            n_misleading = self.add_misleading

        if n_misleading > 0:
            X = np.concatenate([X, np.ones(n_misleading) * (X.max()+1), np.ones(n_misleading) * (X.min()-1)])
            y = np.concatenate([y, np.zeros(n_misleading), np.ones(n_misleading)])

        prior = np.sum(y) / y.size
        weight = y * (1 - prior) + (1 - y) * prior
        self._ir.fit(X, y, sample_weight=weight)

        return self

    def transform(self, X):
        self.p1 = self._ir.transform(X)
        self.p0 = 1 - self.p1
        return to_odds(self.p1)


class FourParameterLogisticCalibrator:
    """
    Calculates a likelihood ratio of a score value, provided it is from one of two distributions.
    Depending on the training data, a 2-, 3- or 4-parameter logistic model is used.
    """
    def __int__(self):
        self.coef_ = None

    def fit(self, X, y):
        # check for negative inf for '1'-labels or inf for '0'-labels
        estimate_c = np.any(np.isneginf(X[y == 1]))
        estimate_d = np.any(np.isposinf(X[y == 0]))

        # define bounds for a and b
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf)]

        if estimate_c and estimate_d:
            # then define 4PL-logistic model
            self.model = self._four_pl_model
            bounds.extend([(10**-10, 1-10**-10), (10**-10, np.inf)])
            LOG.warning("There were -Inf lrs for the same source samples and Inf lrs for the different source samples "
                        ", therefore a 4pl calibrator was fitted.")
        elif estimate_c:
            # then define 3-PL logistic model. Set 'd' to 0
            self.model = partial(self._four_pl_model, d=0)
            # use very small values since limits result in -inf llh
            bounds.append((10**-10, 1-10**-10))
            LOG.warning("There were -Inf lrs for the same source samples, therefore a 3pl calibrator was fitted.")
        elif estimate_d:
            # then define 3-PL logistic model. Set 'c' to 0
            # use bind since 'c' is intermediate variable. In that case partial does not work.
            self.model = Bind(self._four_pl_model, ..., ..., ..., 0, ...)
            # use very small value since limits result in -inf llh
            bounds.append((10**-10, np.inf))
            LOG.warning("There were Inf lrs for the different source samples, therefore a 3pl calibrator was fitted.")
        else:
            # define ordinary logistic model (no regularization, so maximum likelihood estimates)
            self.model = partial(self._four_pl_model, c=0, d=0)
        # define function to minimize
        objective_function = partial(negative_log_likelihood_balanced, X, y, self.model)

        result = minimize(objective_function, np.array([.1] * (2 + estimate_d + estimate_c)),
                          bounds=bounds)
        if not result.success:
            raise Exception("The optimizer did not converge for the calibrator, please check your data.")
        assert result.success
        self.coef_ = result.x

    def transform(self, X):
        """
        Returns the odds ratio.
        """
        return to_odds(self.model(X, *self.coef_))

    @staticmethod
    def _four_pl_model(s, a, b, c, d):
        """
        inputs:
                s: n * 1 np.array of scores
                a,b,c,d,: floats defining 4PL model.
                    a and b are the familiar logistic parameters.
                    c and d respectively floor and ceil the posterior probability
                        the flour probability is c and the ceiling probability is c + (1 - c)/(1 + d)
        output:
                p: n * 1 np.array. Posterior probabilities of succes given each s (and a,b,c,d)
        """
        p = c + ((1 - c) / (1 + d)) * 1 / (1 + np.exp(-a * s - b))
        return p


class DummyCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. No calibration is applied. Instead, the score value is
    interpreted as a posterior probability of the value being sampled from
    class 1.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        self.p0 = (1 - X)
        self.p1 =  X
        return to_odds(self.p1)


class ELUBbounder(BaseEstimator, TransformerMixin):
    """
    Class that, given an LR system, outputs the same LRs as the system but bounded by the Empirical Upper and Lower
    Bounds as described in
    P. Vergeer, A. van Es, A. de Jongh, I. Alberink, R.D. Stoel,
    Numerical likelihood ratios outputted by LR systems are often based on extrapolation:
    when to stop extrapolating?
    Sci. Justics 56 (2016) 482-491

    # MATLAB code from the authors:

    # clear all; close all;
    # llrs_hp=csvread('...');
    # llrs_hd=csvread('...');
    # start=-7; finish=7;
    # rho=start:0.01:finish; theta=10.^rho;
    # nbe=[];
    # for k=1:length(rho)
    #     if rho(k)<0
    #         llrs_hp=[llrs_hp;rho(k)];
    #         nbe=[nbe;(theta(k)^(-1))*mean(llrs_hp<=rho(k))+...
    #             mean(llrs_hd>rho(k))];
    #     else
    #         llrs_hd=[llrs_hd;rho(k)];
    #         nbe=[nbe;theta(k)*mean(llrs_hd>=rho(k))+...
    #             mean(llrs_hp<rho(k))];
    #     end
    # end
    # plot(rho,-log10(nbe)); hold on;
    # plot([start finish],[0 0]);
    # a=rho(-log10(nbe)>0);
    # empirical_bounds=[min(a) max(a)]
    """

    def __init__(self, first_step_calibrator, also_fit_calibrator=True):
        """
        a calibrator should be provided (optionally already fitted to data). This calibrator is called on scores,
        the resulting LRs are then bounded. If also_fit_calibrator, the first step calibrator will be fit on the same
        data used to derive the ELUB bounds
        :param first_step_calibrator: the calibrator to use. Should already have been fitted if also_fit_calibrator is False
        :param also_fit_calibrator: whether to also fit the first step calibrator when calling fit
        """

        self.first_step_calibrator = first_step_calibrator
        self.also_fit_calibrator = also_fit_calibrator
        self._lower_lr_bound = None
        self._upper_lr_bound = None
        if not also_fit_calibrator:
            # check the model was fitted.
            try:
                first_step_calibrator.transform(np.array([0.5]))
            except NotFittedError:
                print('calibrator should have been fit when setting also_fit_calibrator = False!')

    def fit(self, X, y):
        """
        assuming that y=1 corresponds to Hp, y=0 to Hd
        """
        if self.also_fit_calibrator:
            self.first_step_calibrator.fit(X,y)
        lrs  = self.first_step_calibrator.transform(X)

        y = np.asarray(y).squeeze()
        self._lower_lr_bound, self._upper_lr_bound = elub(lrs, y, add_misleading=1)
        return self

    def transform(self, X):
        """
        a transform entails calling the first step calibrator and applying the bounds found
        """
        unadjusted_lrs = np.array(self.first_step_calibrator.transform(X))
        lower_adjusted_lrs = np.where(self._lower_lr_bound < unadjusted_lrs, unadjusted_lrs, self._lower_lr_bound)
        adjusted_lrs = np.where(self._upper_lr_bound > lower_adjusted_lrs, lower_adjusted_lrs, self._upper_lr_bound)
        return adjusted_lrs

    @property
    def p0(self):
        return self.first_step_calibrator.p0

    @property
    def p1(self):
        return self.first_step_calibrator.p1
