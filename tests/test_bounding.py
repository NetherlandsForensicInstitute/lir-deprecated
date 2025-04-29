import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

import lir.bounding as bounding
from lir.data import AlcoholBreathAnalyser, UnboundedLRs
from lir.calibration import LogitCalibrator, IVbounder
from lir.lr import CalibratedScorer
from lir.util import Xn_to_Xy


class TestBounding(unittest.TestCase):

    def test_breath(self):
        lrs, y = AlcoholBreathAnalyser(ill_calibrated=True).sample_lrs()
        bounds = bounding.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((0.1052741, 85.3731634), bounds[:2])

    def test_iv_paper_examples(self):
        llr_threshold = np.arange(-2, 3, 0.001)

        lrs, y = UnboundedLRs(example=4).sample_lrs()
        bounds = bounding.calculate_invariance_bounds(lrs, y, llr_threshold)
        np.testing.assert_almost_equal((0.2382319, 2.7861212), bounds[:2])

        lrs, y = UnboundedLRs(example=5).sample_lrs()
        bounds = bounding.calculate_invariance_bounds(lrs, y, llr_threshold)
        np.testing.assert_almost_equal((0.1412538, 38.1944271), bounds[:2])

    def test_extreme_smallset(self):
        lrs = np.array([np.inf, 0])
        y = np.array([1, 0])
        bounds = bounding.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((0.3335112, 2.9999248), bounds[:2])

    def test_extreme(self):
        lrs = np.array([np.inf, np.inf, np.inf, 0, 0, 0])
        y = np.array([1, 1, 1, 0, 0, 0])
        bounds = bounding.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((0.1429257, 6.9840986), bounds[:2])

    def test_system01(self):
        lrs = np.array([.01, .1, 1, 10, 100])
        bounds_bad = bounding.calculate_invariance_bounds(lrs, np.array([1, 1, 1, 0, 0]))
        bounds_good1 = bounding.calculate_invariance_bounds(lrs, np.array([0, 0, 1, 1, 1]))
        bounds_good2 = bounding.calculate_invariance_bounds(lrs, np.array([0, 0, 0, 1, 1]))

        np.testing.assert_almost_equal((1, 1), bounds_bad[:2])
        np.testing.assert_almost_equal((0.1503142, 3.74973), bounds_good1[:2])
        np.testing.assert_almost_equal((0.2666859, 6.6527316), bounds_good2[:2])

    def test_neutral_smallset(self):
        lrs = np.array([1, 1])
        y = np.array([1, 0])
        bounds = bounding.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 1), bounds[:2])

    def test_bias(self):
        lrs = np.ones(10) * 10
        y = np.concatenate([np.ones(9), np.zeros(1)])
        bounds = bounding.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 1.2647363), bounds[:2])

        lrs = np.concatenate([np.ones(10) * 10, np.ones(1)])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        bounds = bounding.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 3.8106582), bounds[:2])

        lrs = np.concatenate([np.ones(10) * 1000, np.ones(1) * 1.1])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        bounds = bounding.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 3.8106582), bounds[:2])

    def test_bounded_calibrated_scorer(self):

        rng = np.random.default_rng(0)

        X0 = rng.normal(loc=-1, scale=1, size=(1000, 1))
        X1 = rng.normal(loc=+1, scale=1, size=(1000, 1))
        X, y = Xn_to_Xy(X0, X1)

        bounded_calibrated_scorer = CalibratedScorer(LogisticRegression(), IVbounder(LogitCalibrator()))
        bounded_calibrated_scorer.fit(X, y)
        bounds = (bounded_calibrated_scorer.calibrator._lower_lr_bound, bounded_calibrated_scorer.calibrator._upper_lr_bound)
        np.testing.assert_almost_equal((0.0241763, 152.8940706), bounds)

if __name__ == '__main__':
    unittest.main()
