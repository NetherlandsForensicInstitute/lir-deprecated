from sklearn.base import ClassifierMixin, BaseEstimator


class TwoLevelModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        An implementation of the two-level model as outlined in ... .
        As this model uses a different dataset for training than the dataset that is to be used by the calibrator,
        it is not fit for use with lir.CalibratedScorer().
        TODO include paper reference and describe model
        """
        self.dummy_score = None

    def fit(self, X, y):
        """
        Construct the necessary matrices/scores/etc based on test data (X) so that we can predict a score later on.
        Store any calculated parameters in `self`.
        """
        self.dummy_score = 0.5

    def predict_proba(self, X):
        """
        Predict probability-like scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).

        Alternatively we can implement `transform()` if we want to output log-odds instead of probabilities.
        """
        if self.dummy_score is None:
            raise ValueError("The model is not fitted; fit is before you use it for predicting")
        return self.dummy_score

    def predict(self, X):
        """
        To comply with sklearn pipelines, TODO check if this is really necessary
        """
        return self.predict_proba(X)
