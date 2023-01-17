import logging

import numpy as np
import sklearn
import sklearn.mixture
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from .metrics import calculate_lr_statistics
from .util import Xn_to_Xy, LR


LOG = logging.getLogger(__name__)


class EstimatorTransformer(TransformerMixin):
    """
    A wrapper for an estimator to make it behave like a transformer.

    In particular, it implements `transform` by calling `predict_proba` on the underlying estimator.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        return self.estimator.predict_proba(X)[:, 1]

    def __getattr__(self, item):
        return getattr(self.estimator, item)


def _create_transformer(scorer):
    if hasattr(scorer, "transform"):
        return scorer
    elif hasattr(scorer, "predict_proba"):
        return EstimatorTransformer(scorer)
    elif callable(scorer):
        return sklearn.preprocessing.FunctionTransformer(scorer)
    else:
        raise NotImplementedError("`scorer` argument must either be callable or implement at least one of `transform`, `predict_proba`")


class CalibratedScorer:
    """
    LR system which produces calibrated LRs using a scorer and a calibrator.

    The scorer is an object that transforms instance features into a scalar value (a "score"). The scorer may either
    be:
     - an estimator (e.g. `sklearn.linear_model.LogisticRegression`) object which implements `fit` and `predict_proba`;
     - a transformer object which implements `transform` and optionally `fit`; or
     - a callable that takes features as an argument and returns scores.

    The scorer can also be a composite object such as a `sklearn.pipeline.Pipeline`.

    The calibrator is an object that transforms instance scores to LRs.
    """
    def __init__(self, scorer, calibrator):
        """
        Constructor.

        Parameters
        ----------
        scorer an object that transforms features into scores.
        calibrator an object that transforms scores into LRs.
        """
        self.scorer = _create_transformer(scorer)
        self.calibrator = calibrator
        self.pipeline = Pipeline([("scorer", self.scorer), ("calibrator", self.calibrator)])

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_lr(self, X):
        return self.pipeline.transform(X)


class CalibratedScorerCV:
    """
    Variant of `CalibratedScorer` that uses cross-calibration to optimize for small data sets.
    """
    def __init__(self, scorer, calibrator, n_splits):
        self.scorer = _create_transformer(scorer)
        self.calibrator = calibrator
        self.n_splits = n_splits

    def fit(self, X, y):
        kf = sklearn.model_selection.StratifiedKFold(n_splits=self.n_splits, shuffle=True)

        Xcal = np.empty([0, 1])
        ycal = np.empty([0])
        for train_index, cal_index in kf.split(X, y):
            self.scorer.fit(X[train_index], y[train_index])
            p = self.scorer.transform(X[cal_index])
            Xcal = np.append(Xcal, p)
            ycal = np.append(ycal, y[cal_index])
        self.calibrator.fit(Xcal, ycal)

        self.scorer.fit(X, y)
        return self

    def predict_lr(self, X):
        scores = self.scorer.transform(X)
        return self.calibrator.transform(scores)


def calibrate_lr(point, calibrator):
    """
    Calculates a calibrated likelihood ratio (LR).

    P(E|H1) / P(E|H0)
    
    Parameters
    ----------
    point : float
        The point of interest for which the LR is calculated. This point is
        sampled from either class 0 or class 1.
    calibrator : a calibration function
        A calibration function which is fitted for two distributions.

    Returns
    -------
    tuple<3> of float
        A likelihood ratio, with positive values in favor of class 1, and the
        probabilities P(E|H0), P(E|H1).
    """
    lr = calibrator.transform(np.array(point))[0]
    return LR(lr, calibrator.p0[0], calibrator.p1[0])


def apply_scorer(scorer, X):
    return scorer.predict_proba(X)[:,1] # probability of class 1


def scorebased_lr(scorer, calibrator, X0_train, X1_train, X0_calibrate, X1_calibrate, X_disputed):
    """
    Trains a classifier, calibrates the outcome and calculates a LR for a single
    sample.

    P(E|H1) / P(E|H0)

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    scorer : classifier
        A model to be trained. Must support probability output.
    X0_train : numpy array
        Training set of samples from class 0
    X1_train : numpy array
        Training set of samples from class 1
    X0_calibrate : numpy array
        Calibration set of samples from class 0
    X1_calibrate : numpy array
        Calibration set of samlpes from class 1
    X_disputed : numpy array
        Test samples of unknown class

    Returns
    -------
    float
        A likelihood ratio for `X_disputed`
    """
    Xtrain, ytrain = Xn_to_Xy(X0_train, X1_train)
    scorer.fit(Xtrain, ytrain)

    Xcal, ycal = Xn_to_Xy(X0_calibrate, X1_calibrate)
    calibrator.fit(scorer.predict_proba(Xcal)[:,1], ycal)
    scorer = CalibratedScorer(scorer, calibrator)

    return scorer.predict_lr(X_disputed)


def calibrated_cllr(calibrator, class0_calibrate, class1_calibrate, class0_test=None, class1_test=None):
    Xcal, ycal = Xn_to_Xy(class0_calibrate, class1_calibrate)
    calibrator.fit(Xcal, ycal)

    use_calibration_set_for_test = class0_test is None
    if use_calibration_set_for_test:
        class0_test = class0_calibrate
        class1_test = class1_calibrate

    lrs0 = calibrator.transform(class0_test)
    lrs1 = calibrator.transform(class1_test)

    lrs0 = [ LR(*stats) for stats in zip(lrs0, calibrator.p0, calibrator.p1) ]
    lrs1 = [ LR(*stats) for stats in zip(lrs1, calibrator.p0, calibrator.p1) ]

    return calculate_lr_statistics(lrs0, lrs1)


def scorebased_cllr(scorer, calibrator, X0_train, X1_train, X0_calibrate, X1_calibrate, X0_test=None, X1_test=None):
    """
    Trains a classifier on a training set, calibrates the outcome with a
    calibration set, and calculates a LR (likelihood ratio) for all samples in
    a test set. Calculates a Cllr (log likelihood ratio cost) value for the LR
    values. If no test set is provided, the calibration set is used for both
    calibration and testing.

    All numpy arrays have two dimensions, representing samples and features.

    Parameters
    ----------
    scorer : classifier
        A model to be trained. Must support probability output.
    density_function : function
        A density function which is used to deterimine the density of
        a classifier outcome when sampled from either of both classes.
    X0_train : numpy array
        Training set for class 0
    X1_train : numpy array
        Training set for class 1
    X0_calibrate : numpy array
        Calibration set for class 0
    X1_calibrate : numpy array
        Calibration set for class 1
    X0_test : numpy array
        Test set for class 0
    X1_test : numpy array
        Test set for class 1

    Returns
    -------
    float
        A likelihood ratio for `X_test`
    """
    if X0_test is None:
        LOG.debug('scorebased_cllr: training_size: {train0}/{train1}; calibration size: {cal0}/{cal1}'.format(train0=X0_train.shape[0], train1=X1_train.shape[0], cal0=X0_calibrate.shape[0], cal1=X1_calibrate.shape[0]))
    else:
        LOG.debug('scorebased_cllr: training_size: {train0}/{train1}; calibration size: {cal0}/{cal1}; test size: {test0}/{test1}'.format(train0=X0_train.shape[0], train1=X1_train.shape[0], cal0=X0_calibrate.shape[0], cal1=X1_calibrate.shape[0], test0=X0_test.shape[0], test1=X1_test.shape[0]))

    scorer.fit(*Xn_to_Xy(X0_train, X1_train))

    class0_test = apply_scorer(scorer, X0_test) if X0_test is not None else None
    class1_test = apply_scorer(scorer, X1_test) if X0_test is not None else None

    return calibrated_cllr(calibrator, apply_scorer(scorer, X0_calibrate), apply_scorer(scorer, X1_calibrate), class0_test, class1_test)


def scorebased_lr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X_disputed):
    scorer = CalibratedScorerCV(scorer, calibrator, n_splits=n_splits)
    scorer.fit(*Xn_to_Xy(X0_train, X1_train))
    return scorer.predict_lr(X_disputed)


def scorebased_cllr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X0_test, X1_test):
    LOG.debug('scorebased_cllr_kfold: training_size: {train0}/{train1}; test size: {test0}/{test1}'.format(train0=X0_train.shape[0], train1=X1_train.shape[0], test0=X0_test.shape[0], test1=X1_test.shape[0]))

    X_disputed = np.concatenate([X0_test, X1_test])
    lrs = scorebased_lr_kfold(scorer, calibrator, n_splits, X0_train, X1_train, X_disputed)

    assert X0_test.shape[0] + X1_test.shape[0] == len(lrs)
    lrs0 = lrs[:X0_test.shape[0]]
    lrs1 = lrs[X0_test.shape[0]:]

    lrs0 = [ LR(*stats) for stats in zip(lrs0, calibrator.p0, calibrator.p1) ]
    lrs1 = [ LR(*stats) for stats in zip(lrs1, calibrator.p0, calibrator.p1) ]

    return calculate_lr_statistics(lrs0, lrs1)
