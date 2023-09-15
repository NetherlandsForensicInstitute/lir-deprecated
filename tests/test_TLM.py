import csv
import os
import unittest
import numpy as np
from lir.classifiers import TLM_calc_MSwithin

"""
def read_data(path):
    with open(path, 'r') as file:
        r = csv.reader(file)
        next(r)
        data = np.array([float(value) for _, value in r])
    return data
    """



class TestTLM(unittest.TestCase):
    dirname = os.path.dirname(__file__)
    dataZ = np.loadtxt(os.path.join(dirname, 'data/TLM/input/ZDATA.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))

    def test_compare_MSwithin(self):
        MSwithin_R = np.loadtxt(os.path.join(self.dirname, 'data/TLM/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        MSwithin_P = TLM_calc_MSwithin(self.dataZ[:,1:], self.dataZ[:,0])
        np.testing.assert_almost_equal(MSwithin_P, MSwithin_R, decimal=17)

if __name__ == '__main__':
    unittest.main()