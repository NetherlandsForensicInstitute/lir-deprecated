import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_array, check_consistent_length


class IsotonicRegressionInf(IsotonicRegression):
    """
    Sklearn implementation IsotonicRegression throws an error when values are Inf or -Inf when in fact IsotonicRegression
    can handle infinite values. This wrapper around the sklearn implementation of IsotonicRegression prevents the error
    being thrown when Inf or -Inf values are provided.
    """
    def fit(self, X, y, sample_weight=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as :meth:`transform` needs X to interpolate
        new input data.
        """
        check_params = dict(accept_sparse=False, ensure_2d=False, ensure_all_finite=False)
        X = check_array(X, dtype=[np.float64, np.float32], **check_params)
        y = check_array(y, dtype=X.dtype, **check_params)
        check_consistent_length(X, y, sample_weight)

        # Transform y by running the isotonic regression algorithm and
        # transform X accordingly.
        X, y = self._build_y(X, y, sample_weight)

        # It is necessary to store the non-redundant part of the training set
        # on the model to make it possible to support model persistence via
        # the pickle module as the object built by scipy.interp1d is not
        # picklable directly.
        self._necessary_X_, self._necessary_y_ = X, y

        # Build the interpolation function
        self._build_f(X, y)
        return self

    def transform(self, T):
        """Transform new data by linear interpolation

        Parameters
        ----------
        T : array-like of shape (n_samples,)
            Data to transform.

        Returns
        -------
        T_ : array, shape=(n_samples,)
            The transformed data
        """

        if hasattr(self, '_necessary_X_'):
            dtype = self._necessary_X_.dtype
        else:
            dtype = np.float64

        T = check_array(T, dtype=dtype, ensure_2d=False, ensure_all_finite=False)

        if len(T.shape) != 1:
            raise ValueError("Isotonic regression input should be a 1d array")

        # Handle the out_of_bounds argument by clipping if needed
        if self.out_of_bounds not in ["raise", "nan", "clip"]:
            raise ValueError("The argument ``out_of_bounds`` must be in "
                             "'nan', 'clip', 'raise'; got {0}"
                             .format(self.out_of_bounds))

        if self.out_of_bounds == "clip":
            T = np.clip(T, self.X_min_, self.X_max_)

        res = self.f_(T)

        # on scipy 0.17, interp1d up-casts to float64, so we cast back
        res = res.astype(T.dtype)

        return res
