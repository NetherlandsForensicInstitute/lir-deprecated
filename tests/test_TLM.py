import csv
import os
import unittest
import numpy as np
from lir.classifiers import TLM_calc_MSwithin, TLM_calc_means, TLM_calc_h_sq, TLM_calc_T0, TLM_calc_U, TLM_calc_mu_h


class TestTLM(unittest.TestCase):
    dirname = os.path.dirname(__file__)
    dataZ = np.loadtxt(os.path.join(dirname, 'data/TLM/input/ZDATA.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))
    dataX = np.loadtxt(os.path.join(dirname, 'data/TLM/input/XDATA.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 11))
    dataY = np.loadtxt(os.path.join(dirname, 'data/TLM/input/YDATA.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))
    # Calculate the parameters from the fit in python (needed for the predict functions)
    MSwithin_P = TLM_calc_MSwithin(dataZ[:, 1:], dataZ[:, 0])
    means_z_P = TLM_calc_means(dataZ[:, 1:], dataZ[:, 0])
    h_sq_P = TLM_calc_h_sq(dataZ[:, 1:], dataZ[:, 0])
    T0_P = TLM_calc_T0(dataZ[:, 1:], dataZ[:, 0])

    def test_MSwithin(self):
        MSwithin_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        MSwithin_P = TLM_calc_MSwithin(self.dataZ[:,1:], self.dataZ[:,0])
        np.testing.assert_almost_equal(MSwithin_P, MSwithin_R, decimal=17)

    def test_means(self):
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
    def test_T0(self):
        T0_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/T0.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        T0_P = TLM_calc_T0(self.dataZ[:, 1:], self.dataZ[:, 0])
        np.testing.assert_almost_equal(T0_P, T0_R, decimal=15)


    def test_U_h0(self):
        U_h0_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/U_h0.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        U_h0_P =  TLM_calc_U(self.dataY[[0, 1, 2], 1:], self.dataX, self.MSwithin_P, self.h_sq_P, self.T0_P)[0]
        np.testing.assert_almost_equal(U_h0_P, U_h0_R, decimal=16)

    def test_U_hx(self):
        U_hx_R  = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/U_hx.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        U_hx_P = TLM_calc_U(self.dataY[[0, 1, 2], 1:], self.dataX, self.MSwithin_P, self.h_sq_P, self.T0_P)[1]
        np.testing.assert_almost_equal(U_hx_P, U_hx_R, decimal=16)

    def test_U_hn(self):
        U_hn_R  = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/U_hn.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        U_hn_P = TLM_calc_U(self.dataY[[0, 1, 2], 1:], self.dataX, self.MSwithin_P, self.h_sq_P, self.T0_P)[2]
        np.testing.assert_almost_equal(U_hn_P, U_hn_R, decimal=15)

    def test_mu_h(self):
        mu_h_R  = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/mu_h.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        mu_h_P = TLM_calc_mu_h(self.dataX, self.MSwithin_P, self.T0_P, self.h_sq_P, self.dataZ[:, 1:], self.dataZ[:, 0])
        np.testing.assert_almost_equal(mu_h_P, mu_h_R, decimal=14)

if __name__ == '__main__':
    unittest.main()