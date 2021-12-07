import numpy as np
import unittest
import warnings

from context import lir
assert lir  # so import optimizer doesn't remove the line above

from sklearn.linear_model import LogisticRegression

from lir.calibration import FractionCalibrator, ScalingCalibrator
from lir import metrics
from lir.lr import scorebased_cllr
from lir.util import Xn_to_Xy


warnings.simplefilter("error")


class TestLR(unittest.TestCase):
    def test_fraction_calibrator(self):
        points_h0 = np.array([ 1, 2, 4, 8 ])
        points_h1 = np.array([ 2, 6, 8, 9 ])
        p0 = np.array([1., 1., .75, .5, .5 , .25, .25, .25, .25, 0., 0.])
        p1 = np.array([0., 0., .25, .25, .25, .25, .5, .5, .75, 1, 1])

        cal = FractionCalibrator(value_range=[0,10])
        cal.fit(*Xn_to_Xy(points_h0, points_h1))

        lr = cal.transform(np.arange(11))
        np.testing.assert_almost_equal(cal.p0, p0)
        np.testing.assert_almost_equal(cal.p1, p1)
        with np.errstate(divide='ignore'):
            np.testing.assert_almost_equal(lr, p1/p0)

    def test_calculate_cllr(self):
        self.assertAlmostEqual(1, metrics.cllr(*Xn_to_Xy([1, 1], [1, 1])))
        self.assertAlmostEqual(2, metrics.cllr(*Xn_to_Xy([3.]*2, [1/3.]*2)))
        self.assertAlmostEqual(2, metrics.cllr(*Xn_to_Xy([3.]*20, [1/3.]*20)))
        self.assertAlmostEqual(0.4150374992788437, metrics.cllr(*Xn_to_Xy([1/3.]*2, [3.]*2)))
        self.assertAlmostEqual(0.7075187496394219, metrics.cllr(*Xn_to_Xy([1/3.]*2, [1])))
        self.assertAlmostEqual(0.507177646488535, metrics.cllr(*Xn_to_Xy([1/100.]*100, [1])))
        self.assertAlmostEqual(0.5400680236656377, metrics.cllr(*Xn_to_Xy([1/100.]*100 + [100], [1])))
        self.assertAlmostEqual(0.5723134914863265, metrics.cllr(*Xn_to_Xy([1/100.]*100 + [100]*2, [1])))
        self.assertAlmostEqual(0.6952113122368764, metrics.cllr(*Xn_to_Xy([1/100.]*100 + [100]*6, [1])))
        self.assertAlmostEqual(1.0000000000000000, metrics.cllr(*Xn_to_Xy([1], [1])))
        self.assertAlmostEqual(1.0849625007211563, metrics.cllr(*Xn_to_Xy([2], [2]*2)))
        self.assertAlmostEqual(1.6699250014423126, metrics.cllr(*Xn_to_Xy([8], [8]*8)))

    def test_extreme_cllr(self):
        self.assertEqual(np.inf, metrics.cllr(*Xn_to_Xy([np.inf, 1], [1, 1])))
        self.assertEqual(np.inf, metrics.cllr(*Xn_to_Xy([np.inf, 0], [1, 1])))
        self.assertEqual(np.inf, metrics.cllr(*Xn_to_Xy([1, 1], [0, 1])))
        self.assertAlmostEqual(.5, metrics.cllr(*Xn_to_Xy([0, 0], [1, 1])))
        self.assertAlmostEqual(.5, metrics.cllr(*Xn_to_Xy([1, 1], [np.inf, np.inf])))
        self.assertAlmostEqual(0, metrics.cllr(*Xn_to_Xy([0, 0], [np.inf, np.inf])))
        self.assertAlmostEqual(np.inf, metrics.cllr(*Xn_to_Xy([np.inf, np.inf], [0, 0])))
        self.assertEqual(np.inf, metrics.cllr(*Xn_to_Xy([1], [1.e-317]))) # value near zero for which 1/value causes an overflow

    def test_illegal_cllr(self):
        self.assertTrue(np.isnan(metrics.cllr(*Xn_to_Xy([np.nan, 1], [1, 1]))))
        self.assertTrue(np.isnan(metrics.cllr(*Xn_to_Xy([1, 1], [1, np.nan]))))
        self.assertTrue(np.isnan(metrics.cllr(*Xn_to_Xy([np.nan, np.nan], [np.nan, np.nan]))))

    def test_classifier_cllr(self):
        np.random.seed(0)
        clf = LogisticRegression(solver='lbfgs')
        cal = ScalingCalibrator(FractionCalibrator())

        prev_cllr = 1
        for i in range(1, 10):
            X0 = np.random.normal(loc=[-1]*3, scale=.1, size=(i, 3))
            X1 = np.random.normal(loc=[1]*3, scale=.1, size=(i, 3))
            cllr = scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr
            self.assertLess(cllr, prev_cllr)
            prev_cllr = cllr

        cal = FractionCalibrator()

        X0 = np.random.normal(loc=[-1]*3, size=(100, 3))
        X1 = np.random.normal(loc=[1]*3, size=(100, 3))
        self.assertAlmostEqual(.13257776120905165, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X0 = np.random.normal(loc=[-.5]*3, size=(100, 3))
        X1 = np.random.normal(loc=[.5]*3, size=(100, 3))
        self.assertAlmostEqual(.6514624971651655, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X0 = np.random.normal(loc=[0]*3, size=(100, 3))
        X1 = np.random.normal(loc=[0]*3, size=(100, 3))
        self.assertAlmostEqual(1.3502413785060203, scorebased_cllr(clf, cal, X0, X1, X0, X1).cllr)

        X = np.random.normal(loc=[0]*3, size=(400, 3))
        self.assertAlmostEqual(1.3742926488365286, scorebased_cllr(clf, cal, X[:100], X[100:200], X[200:300], X[300:400]).cllr)


if __name__ == '__main__':
    unittest.main()
