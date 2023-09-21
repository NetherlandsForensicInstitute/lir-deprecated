class TwoLevelModel:
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

    def transform(self, X):
        """
        Predict log-odds scores, making use of the parameters constructed during `self.fit()` (which should
        now be stored in `self`).
        """
        if self.dummy_score is None:
            raise ValueError("The model is not fitted; fit it before you use it for predicting")
        return self.dummy_score

