import logging


LOG = logging.getLogger(__name__)


class CalibratedScorer:
    def __init__(self, scorer, calibrator):
        self.scorer = scorer
        self.calibrator = calibrator

    def fit(self, X, y):
        self.fit_scorer(X, y)
        self.fit_calibrator(X, y)

    def fit_scorer(self, X, y):
        self.scorer.fit(X, y)

    def fit_calibrator(self, X, y):
        p = self.scorer.predict_proba(X)
        self.calibrator.fit(p, y)

    def predict(self, X):
        return self.scorer.predict(X)

    def predict_lr(self, X):
        X = self.scorer.predict_proba(X)
        return self.calibrator.transform(X)
