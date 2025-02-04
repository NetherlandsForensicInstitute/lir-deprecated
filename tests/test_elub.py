import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

import lir.bayeserror
from lir.data import AlcoholBreathAnalyser
from lir.calibration import KDECalibrator, ELUBbounder
from lir.lr import CalibratedScorer
from lir.util import Xn_to_Xy


class TestElub(unittest.TestCase):

    def test_breath(self):
        lrs, y = AlcoholBreathAnalyser(ill_calibrated=True).sample_lrs()
        bounds = lir.bayeserror.elub(lrs, y, add_misleading=1)
        np.testing.assert_almost_equal((0.11051160265422605, 80.42823031359497), bounds)

    def test_extreme_smallset(self):
        lrs = np.array([np.inf, 0])
        y = np.array([1, 0])
        bounds = lir.bayeserror.elub(lrs, y, add_misleading=1)
        np.testing.assert_almost_equal((1, 1), bounds)

    def test_extreme(self):
        lrs = np.array([np.inf, np.inf, np.inf, 0, 0, 0])
        y = np.array([1, 1, 1, 0, 0, 0])
        bounds = lir.bayeserror.elub(lrs, y, add_misleading=1)
        np.testing.assert_almost_equal((0.3362165, 2.9742741), bounds)

    def test_system01(self):
        lrs = np.array([.01, .1, 1, 10, 100])
        bounds_bad = lir.bayeserror.elub(lrs, np.array([1, 1, 1, 0, 0]), add_misleading=1)
        bounds_good1 = lir.bayeserror.elub(lrs, np.array([0, 0, 1, 1, 1]), add_misleading=1)
        bounds_good2 = lir.bayeserror.elub(lrs, np.array([0, 0, 0, 1, 1]), add_misleading=1)

        np.testing.assert_almost_equal((1, 1), bounds_bad)
        np.testing.assert_almost_equal((0.3771282, 1.4990474), bounds_good1)
        np.testing.assert_almost_equal((0.6668633, 2.6507161), bounds_good2)

    def test_neutral_smallset(self):
        lrs = np.array([1, 1])
        y = np.array([1, 0])
        bounds = lir.bayeserror.elub(lrs, y, add_misleading=1)
        np.testing.assert_almost_equal((1, 1), bounds)


    def test_bias(self):
        lrs = np.ones(10) * 10
        y = np.concatenate([np.ones(9), np.zeros(1)])
        np.testing.assert_almost_equal((1, 1), lir.bayeserror.elub(lrs, y, add_misleading=1))

        lrs = np.concatenate([np.ones(10) * 10, np.ones(1)])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        np.testing.assert_almost_equal((1, 1.8039884), lir.bayeserror.elub(lrs, y, add_misleading=1))

        lrs = np.concatenate([np.ones(10) * 1000, np.ones(1) * 1.1])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        np.testing.assert_almost_equal((1, 1), lir.bayeserror.elub(lrs, y, add_misleading=1))

    def test_bounded_calibrated_scorer(self):

        rng = np.random.default_rng(0)

        X0 = rng.normal(loc=-1, scale=1, size=(1000, 1))
        X1 = rng.normal(loc=+1, scale=1, size=(1000, 1))
        X, y = Xn_to_Xy(X0, X1)

        bounded_calibrated_scorer = CalibratedScorer(LogisticRegression(), ELUBbounder(KDECalibrator(bandwidth=(1, 1))))
        bounded_calibrated_scorer.fit(X, y)
        bounds = (bounded_calibrated_scorer.calibrator._lower_lr_bound, bounded_calibrated_scorer.calibrator._upper_lr_bound)
        np.testing.assert_almost_equal((0.063251, 17.6256669), bounds)

if __name__ == '__main__':
    unittest.main()
