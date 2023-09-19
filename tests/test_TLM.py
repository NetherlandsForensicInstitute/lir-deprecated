import csv
import os
import unittest
import numpy as np
from lir.classifiers import TLM_calc_MSwithin, TLM_calc_means, TLM_calc_h_sq

class TestTLM(unittest.TestCase):
    dirname = os.path.dirname(__file__)
    dataZ = np.loadtxt(os.path.join(dirname, 'data/TLM/input/ZDATA.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))

    def test_MSwithin(self):
        MSwithin_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        MSwithin_P = TLM_calc_MSwithin(self.dataZ[:,1:], self.dataZ[:,0])
        np.testing.assert_almost_equal(MSwithin_P, MSwithin_R, decimal=17)

    def test_means_z(self):
        means_z_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/means_z.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        means_z_R = means_z_R.transpose()
        means_z_P = TLM_calc_means(self.dataZ[:, 1:], self.dataZ[:, 0])
        np.testing.assert_almost_equal(means_z_P, means_z_R, decimal=14)

    def test_h_sq(self):
        h_sq_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/h2.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        h_sq_P = TLM_calc_h_sq(self.dataZ[:, 1:], self.dataZ[:, 0])
        np.testing.assert_almost_equal(h_sq_P, h_sq_R, decimal=16)


if __name__ == '__main__':
    unittest.main()