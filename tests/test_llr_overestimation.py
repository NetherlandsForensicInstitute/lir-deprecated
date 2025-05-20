import numpy as np
import unittest

from lir.metrics import llr_overestimation
from lir.llr_overestimation import calc_llr_overestimation

class TestLLROverestimation(unittest.TestCase):
    def test_llr_overestimation_metric(self):

        lnlr_mean = np.log(10 ** 2)
        lnlr_sigma = np.sqrt(2 * lnlr_mean)

        # small perfect system
        rng = np.random.default_rng(42)
        lr_h1 = np.exp(rng.normal(+lnlr_mean, lnlr_sigma, 100))
        lr_h2 = np.exp(rng.normal(-lnlr_mean, lnlr_sigma, 100))
        lrs = np.concatenate((lr_h1, lr_h2))
        y = np.concatenate((np.ones(len(lr_h1)), np.zeros(len(lr_h2))))
        np.testing.assert_almost_equal(0.2473928, llr_overestimation(lrs, y, seed=42))

        # big perfect system
        rng = np.random.default_rng(42)
        lr_h1 = np.exp(rng.normal(+lnlr_mean, lnlr_sigma, 1000))
        lr_h2 = np.exp(rng.normal(-lnlr_mean, lnlr_sigma, 1000))
        lrs = np.concatenate((lr_h1, lr_h2))
        y = np.concatenate((np.ones(len(lr_h1)), np.zeros(len(lr_h2))))
        np.testing.assert_almost_equal(0.0736284, llr_overestimation(lrs, y, seed=42))

        # big shifted system
        rng = np.random.default_rng(42)
        lr_h1 = np.exp(rng.normal(+lnlr_mean, lnlr_sigma, 1000) + np.log(10 ** 1))
        lr_h2 = np.exp(rng.normal(-lnlr_mean, lnlr_sigma, 1000) + np.log(10 ** 1))
        lrs = np.concatenate((lr_h1, lr_h2))
        y = np.concatenate((np.ones(len(lr_h1)), np.zeros(len(lr_h2))))
        np.testing.assert_almost_equal(0.9678462, llr_overestimation(lrs, y, seed=42))

        # big exaggerating system
        rng = np.random.default_rng(42)
        lr_h1 = np.exp(rng.normal(+lnlr_mean, lnlr_sigma, 1000) * 3)
        lr_h2 = np.exp(rng.normal(-lnlr_mean, lnlr_sigma, 1000) * 3)
        lrs = np.concatenate((lr_h1, lr_h2))
        y = np.concatenate((np.ones(len(lr_h1)), np.zeros(len(lr_h2))))
        np.testing.assert_almost_equal(2.1476174, llr_overestimation(lrs, y, seed=42))

    def test_llr_overestimation_interval(self):

        lnlr_mean = np.log(10 ** 2)
        lnlr_sigma = np.sqrt(2 * lnlr_mean)

        # small system
        rng = np.random.default_rng(42)
        lr_h1 = np.exp(rng.normal(+lnlr_mean, lnlr_sigma, 100))
        lr_h2 = np.exp(rng.normal(-lnlr_mean, lnlr_sigma, 100))
        lrs = np.concatenate((lr_h1, lr_h2))
        y = np.concatenate((np.ones(len(lr_h1)), np.zeros(len(lr_h2))))
        llr_overestimation_interval = calc_llr_overestimation(lrs, y, seed=42)[2]
        interval_width = llr_overestimation_interval[:,2] - llr_overestimation_interval[:,0]
        np.testing.assert_almost_equal(1.5305846, np.nanmean(np.abs(interval_width)))

        # big system
        rng = np.random.default_rng(42)
        lr_h1 = np.exp(rng.normal(+lnlr_mean, lnlr_sigma, 1000))
        lr_h2 = np.exp(rng.normal(-lnlr_mean, lnlr_sigma, 1000))
        lrs = np.concatenate((lr_h1, lr_h2))
        y = np.concatenate((np.ones(len(lr_h1)), np.zeros(len(lr_h2))))
        llr_overestimation_interval = calc_llr_overestimation(lrs, y, seed=42)[2]
        interval_width = llr_overestimation_interval[:,2] - llr_overestimation_interval[:,0]
        np.testing.assert_almost_equal(0.7807411, np.nanmean(np.abs(interval_width)))

if __name__ == '__main__':
    unittest.main()
