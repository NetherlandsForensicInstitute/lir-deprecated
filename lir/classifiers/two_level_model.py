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

    def fit(self, X, y):
        """
        Construct the necessary matrices/scores/etc based on test data (X) so that we can predict a score later on.
        Store any calculated parameters in `self`.
        """
        self.model_fitted = True

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

