import numpy as np
import unittest

from lir.metrics import devpav, _calcsurface, _devpavcalculator
from lir.util import Xn_to_Xy

class TestDevPAV(unittest.TestCase):
    def test_devpav_error(self):
        lrs = np.ones(10)
        y = np.concatenate([np.ones(10)])
        with self.assertRaises(ValueError):
            devpav(lrs, y)

    def test_devpav(self):
        # naive system
        lrs = np.ones(10)
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertEqual(devpav(lrs, y), 0)

        # badly calibrated naive system
        lrs = 2*np.ones(10)
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertEqual(devpav(lrs, y), np.log10(2))

        # infinitely bad calibration
        lrs = np.array([5, 5, 5, .2, .2, .2, np.inf])
        y = np.concatenate([np.ones(3), np.zeros(4)])
        self.assertEqual(devpav(lrs, y), np.inf)

        # binary system
        lrs = np.array([5, 5, 5, .2, 5, .2, .2, .2])
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y), (np.log10(5)-np.log10(3))/2)

        # somewhat normal
        lrs = np.array([6, 5, 5, .2, 5, .2, .2, .1])
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(lrs, y), (np.log10(5)-np.log10(2))/2)

        # test on dummy data 3 #######################
        lrs_same = (0.1, 100)
        lrs_dif = (10 ** -2, 10)
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertEqual(devpav(lrs, y), 0.5)


class TestDevpavcalculator(unittest.TestCase):
    def test_devpavcalculator(self):
        ## four tests on pathological PAV-transforms
        # 1 of 4: test on data where PAV-tranform has a horizontal line starting at log(X) = -Inf
        lrs_same = (0, 1, 10**3)
        lrs_dif = (0.001, 2, 10**2)
        PAVresult = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667, np.inf])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertEqual(_devpavcalculator(lrs, PAVresult, y), np.inf)


        # 2 of 4: test on data where PAV-tranform has a horizontal line ending at log(X) = Inf
        lrs_same = (0.01, 1, 10**2)
        lrs_dif = (0.001, 2, float('inf'))
        PAVresult = np.array([0.,  1.5, 1.5, 1.5, 1.5, 1.5])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertEqual(_devpavcalculator(lrs, PAVresult, y), np.inf)


        # 3 of 4: test on data where PAV-tranform has a horizontal line starting at log(X) = -Inf, and another one ending at log(X) = Inf
        lrs_same = (0, 1, 10**3, 10**3, 10**3, 10**3)
        lrs_dif = (0.001, 2, float('inf'))
        PAVresult = np.array([0.5, 0.5, 2, 0.5, 0.5, 2,  2,  2,  2])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertEqual(_devpavcalculator(lrs, PAVresult, y), np.inf)


        # 4 of 4: test on data where lrs_same and lrs_dif are completely seperated (and PAV result is a vertical line)
        lrs_same = (10**4, 10**5, float('inf'))
        lrs_dif = (0, 1, 10**3)
        PAVresult = np.array([0, 0, 0, float('inf'), float('inf'), float('inf')])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertTrue(np.isnan(_devpavcalculator(lrs, PAVresult, y)))

        ### tests on ordinary data

        #test on dummy data. This PAV-transform is parallel to the identity line
        lrs_same = (1, 10**3)
        lrs_dif = (0.1, 10)
        PAVresult = np.array([0, 1, 1, float('inf')])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertAlmostEqual(_devpavcalculator(lrs, PAVresult, y), 0.5)


        #test on dummy data 2, this PAV-transform crosses the identity line
        lrs_same = (0.1, 100, 10**3)
        lrs_dif = (10**-3, 10**-2, 10)
        fakePAVresult = np.array([0, 10**-3, 10**2, 10**-2, 10**2, float('inf')])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertAlmostEqual(_devpavcalculator(lrs, fakePAVresult, y), (1 + 2 * (0.5 * 2 * 1 - 0.5 * 1 * 1) + 0.5)/4)


        # test on dummy data 3, this PAV-transform is finite
        lrs_same = (0.1, 100)
        lrs_dif = (10**-2, 10)
        fakePAVresult = np.array([10**-3, 10**2, 10**-2, 10**2])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertAlmostEqual(_devpavcalculator(lrs, fakePAVresult, y), (1 + 2 * (0.5 * 2 * 1 - 0.5 * 1 * 1) + 0.5)/4)


    def test_calcsurface(self):
        # tests for the _calcsurface function

        # the line segment is parallel to the identity line
        c1 = (4, 1)
        c2 = (10, 7)
        self.assertAlmostEqual(_calcsurface(c1, c2), 18)

        # 2nd possibility (situation 1 of 2 in code below, the intersection with the identity line is within the line segment, y1 < x1)
        c1 = (-1, -2)
        c2 = (0, 3)
        self.assertAlmostEqual(_calcsurface(c1, c2), 1.25)

        # 3rd possibility (situation 2 of 2 in code below, the intersection with the identity line is within the line segment, y1 >= x1)
        c1 = (0, 3)
        c2 = (10, 4)
        self.assertAlmostEqual(_calcsurface(c1, c2), 25)

        # 5th possibility (situation 1 of 4 in code below, both coordinates are below the identity line, intersection with identity line on left hand side)
        c1 = (-1, -2)
        c2 = (0, -1.5)
        self.assertAlmostEqual(_calcsurface(c1, c2), 1.25)

        # 6th possibility (situation 2 van 4 in code below, both coordinates are above the identity line, intersection with identity line on right hand side)
        c1 = (1, 2)
        c2 = (1.5, 2)
        self.assertAlmostEqual(_calcsurface(c1, c2), 0.375)

        # 7th possibility (situation 2 of 4 in code below, both coordinates are above the identity line, intersection with identity line on left hand side)
        c1 = (1, 2)
        c2 = (2, 4)
        self.assertAlmostEqual(_calcsurface(c1, c2), 1.5)

        # 8th possibility (situation 3 of 4 in code below, both coordinates are below the identity line, intersection with identity line on right hand side)
        c1 = (-1, -2)
        c2 = (0, -0.5)
        self.assertAlmostEqual(_calcsurface(c1, c2), 0.75)


        #test with negative slope
        c1 = (1, 4)
        c2 = (2, 2)
        #self.assertEqual(_calcsurface(c1, c2), None)
        with self.assertRaises(Exception) as context:
            _calcsurface(c1, c2)
