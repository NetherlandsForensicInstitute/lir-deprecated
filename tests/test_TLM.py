import os
import unittest
import numpy as np
from lir.classifiers import TLM_calc_MSwithin

def read_data(path):
    with open(path, 'r') as file:
        r = csv.reader(file)
        next(r)
        data = np.array([float(value) for _, value in r])
    return data

dirname = os.path.dirname(__file__)

def test_compare_MSwithin(self):
    MSwithin_R = read_data(os.path.join(dirname, 'data/TLM/R_output/MSwithin.csv'))
    dataZ = read_data(os.path.join(dirname, 'data/TLM/input/ZDATA.csv'))
    MSwithin_P = TLM_calc_MSwithin(dataZ)
    np.testing.assert_equal(MSwithin_P, MSwithin_R)
