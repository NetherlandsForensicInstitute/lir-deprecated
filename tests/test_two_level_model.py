import unittest
import os
import numpy as np

from lir.classifiers.two_level_model import TwoLevelModel


class TestTwoLevelModel(unittest.TestCase):

    two_level_model = TwoLevelModel()

    dirname = os.path.dirname(__file__)
    data_train = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/input/train_data.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))

    def test_mean_covariance_within(self):
        mean_cov_within_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        mean_cov_within_P = self.two_level_model._fit_mean_covariance_within(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(mean_cov_within_P, mean_cov_within_R, decimal=17)

    def test_means_train(self):
        means_train_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/means_z.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        means_train_R = means_train_R.transpose()
        means_train_P = self.two_level_model._fit_means_per_source(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(means_train_P, means_train_R, decimal=14)

    def test_kernel_bandwidth_sq(self):
        kernel_bandwidth_sq_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/h2.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        kernel_bandwidth_sq_P = self.two_level_model._fit_kernel_bandwidth_squared(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(kernel_bandwidth_sq_P, kernel_bandwidth_sq_R, decimal=16)

    def test_between_covars(self):
        between_covars_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/T0.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        mean_cov_within_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        self.two_level_model.mean_covars = mean_cov_within_R
        self.two_level_model.model_fitted = True
        between_covars_P = self.two_level_model._fit_between_covariance(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(between_covars_P, between_covars_R, decimal=15)

if __name__ == '__main__':
    unittest.main()
