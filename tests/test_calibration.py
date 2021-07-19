import numpy as np
import unittest
import warnings

from context import lir

from lir.calibration import IsotonicCalibrator
from lir.util import Xn_to_Xy, Xy_to_Xn
import math


warnings.simplefilter("error")


def _cllr(lr0, lr1):
    with np.errstate(divide='ignore'):
        cllr0 = np.mean(np.log2(1 + lr0))
        cllr1 = np.mean(np.log2(1 + 1/lr1))
        return .5 * (cllr0 + cllr1)


def _pdf(X, mu, sigma):
    return np.exp(-np.power(X - mu, 2) / (2*sigma*sigma)) / math.sqrt(2*math.pi*sigma*sigma)


class TestIsotonicRegression(unittest.TestCase):
    def test_lr_1(self):
        score_class0 = np.arange(0, 1, .1)
        score_class1 = np.arange(0, 1, .1)
        X, y = Xn_to_Xy(score_class0, score_class1)
        irc = IsotonicCalibrator()
        lr0, lr1 = Xy_to_Xn(irc.fit_transform(X, y), y)
        self.assertEqual(score_class0.shape, lr0.shape)
        self.assertEqual(score_class1.shape, lr1.shape)
        np.testing.assert_almost_equal(lr0, [1.]*lr0.shape[0])
        np.testing.assert_almost_equal(lr1, [1.]*lr1.shape[0])

    def run_cllrmin(self, lr0, lr1, places=7):
        lr0 = np.array(lr0)
        lr1 = np.array(lr1)
        X, y = Xn_to_Xy(lr0, lr1)
        cllr = _cllr(lr0, lr1)

        irc = IsotonicCalibrator()
        lrmin0, lrmin1 = Xy_to_Xn(irc.fit_transform(X / (X + 1), y), y)
        cllrmin = _cllr(lrmin0, lrmin1)

        self.assertAlmostEqual(cllr, cllrmin, places=places)

    def test_cllrmin(self):
        self.run_cllrmin([1]*10, [1]*10)
        self.run_cllrmin([1], [1]*10)
        self.run_cllrmin([4, .25, .25, .25, .25, 1], [4, 4, 4, 4, .25, 1])

        #np.random.seed(0)
        X0 = np.random.normal(loc=0, scale=1, size=(40000,))
        X1 = np.random.normal(loc=1, scale=1, size=(40000,))
        lr0 = _pdf(X0, 1, 1) / _pdf(X0, 0, 1)
        lr1 = _pdf(X1, 1, 1) / _pdf(X1, 0, 1)
        self.run_cllrmin(lr0, lr1, places=2)
        self.run_cllrmin(lr0, lr1[:30000], places=2)

    def test_lr_almost_1(self):
        score_class0 = np.arange(0, 1, .1)
        score_class1 = np.arange(.05, 1.05, .1)
        X, y = Xn_to_Xy(score_class0, score_class1)
        irc = IsotonicCalibrator()
        lr0, lr1 = Xy_to_Xn(irc.fit_transform(X, y), y)
        self.assertEqual(score_class0.shape, lr0.shape)
        self.assertEqual(score_class1.shape, lr1.shape)
        np.testing.assert_almost_equal(lr0, np.concatenate([[0], [1.]*(lr0.shape[0]-1)]))
        np.testing.assert_almost_equal(lr1, np.concatenate([[1.]*(lr1.shape[0]-1), [np.inf]]))


if __name__ == '__main__':
    unittest.main()
