import unittest
import os
import numpy as np

from lir.classifiers.two_level_model import TwoLevelModel


class TestTwoLevelModel(unittest.TestCase):

    two_level_model = TwoLevelModel()

    dirname = os.path.dirname(__file__)
    data_train = np.loadtxt(os.path.join(dirname, 'data/TLM/input/ZDATA.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))

    def test_mean_covariance_within(self):
        mean_cov_within_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        mean_cov_within_P = self.two_level_model.fit_mean_covariance_within(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(mean_cov_within_P, mean_cov_within_R, decimal=17)

    def test_means_train(self):
        means_train_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/means_z.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        means_train_R = means_train_R.transpose()
        means_train_P = self.two_level_model.fit_means_per_source(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(means_train_P, means_train_R, decimal=14)

if __name__ == '__main__':
    unittest.main()
