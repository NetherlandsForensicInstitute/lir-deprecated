import numpy as np
import unittest

import lir.bayeserror
from lir.data import AlcoholBreathAnalyser


class TestElub(unittest.TestCase):
    """
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
        """

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


if __name__ == '__main__':
    unittest.main()
