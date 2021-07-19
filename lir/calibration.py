import logging
import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from .bayeserror import elub
from .regression import IsotonicRegressionInf
from .util import Xy_to_Xn, to_odds

LOG = logging.getLogger(__name__)


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


class KDECalibrator(BaseEstimator, TransformerMixin, ):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, bandwidth: Optional[Union[float, Tuple[Optional[float], Optional[float]]]] = None):
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
        self.bandwidth: Tuple[Optional[float], Optional[float]] = \
            self._parse_bandwidth(bandwidth)
        self._kde0: Optional[KernelDensity] = None
        self._kde1: Optional[KernelDensity] = None

    @staticmethod
    def bandwidth_silverman(X):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0

        std = np.std(X)
        if std == 0:
            # can happen eg if std(X) = 0
            warnings.warn('silverman bandwidth cannot be calculated if standard deviation is 0', RuntimeWarning)
            LOG.info('found a silverman bandwidth of 0 (using dummy value)')
            std = 1

        v = math.pow(std, 5) / len(X) * 4. / 3
        return math.pow(v, .2)

    @staticmethod
    def bandwidth_scott(X):
        """
        Not implemented.
        """
        raise

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)

        bandwidth0 = self.bandwidth[0] or self.bandwidth_silverman(X0)
        bandwidth1 = self.bandwidth[1] or self.bandwidth_silverman(X1)

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

    @staticmethod
    def _parse_bandwidth(bandwidth: Optional[Union[float, Tuple[float, float]]]) \
            -> Tuple[Optional[float], Optional[float]]:
        """
        Returns bandwidth as a tuple of two (optional) floats.
        Extrapolates a single bandwidth
        :param bandwidth: provided bandwidth
        :return: bandwidth used for kde0, bandwidth used for kde1
        """
        if bandwidth is None:
            return None, None
        elif isinstance(bandwidth, float):
            return bandwidth, bandwidth
        elif len(bandwidth) == 2:
            return bandwidth
        else:
            raise ValueError('Invalid input for bandwidth')


class LogitCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses logistic regression for interpolation.
    """

    def fit(self, X, y):
        X = X.reshape(-1, 1)
        self._logit = LogisticRegression(class_weight='balanced')
        self._logit.fit(X, y)
        return self

    def transform(self, X):
        X = self._logit.predict_proba(X.reshape(-1, 1))[:, 1]  # probability of class 1
        self.p0 = (1 - X)
        self.p1 = X
        return self.p1 / self.p0


class GaussianCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses a gaussian mixture model for interpolation.
    """

    def fit(self, X, y):
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        self._model0 = GaussianMixture().fit(X0)
        self._model1 = GaussianMixture().fit(X1)
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