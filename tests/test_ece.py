#!/usr/bin/env python3

import numpy as np
import unittest

import lir.ece
from lir.data import AlcoholBreathAnalyser


class TestECE(unittest.TestCase):
    def _compare_ece_cllr(self, lrs, y):
        ece = lir.ece.calculate_ece(lrs, y, np.array([.5, .5]))
        cllr = lir.metrics.cllr(lrs, y)
        np.testing.assert_almost_equal(ece[0], cllr)


    def test_cllr(self):
        data = [
            AlcoholBreathAnalyser(ill_calibrated=True).sample_lrs(),
            (np.array([.1, 10, np.inf]), np.array([0, 1, 1])),
            (np.array([.1, 10, np.inf]), np.array([0, 1, 0])),
            (np.array([.1, 10, 0]), np.array([0, 1, 1])),
            (np.array([.1, 10, 0]), np.array([0, 1, 0])),
        ]

        for lrs, y in data:
            self._compare_ece_cllr(lrs, y)


    def test_invalid(self):
        data = [
            (np.array([.1, 10, np.nan]), np.array([0, 1, 1]), "invalid input for LR values"),
            (np.array([.1, 10, -1]), np.array([0, 1, 0]), "invalid input for LR values"),
        ]

        for lrs, y, msg in data:
            with self.assertRaises(AssertionError) as context:
                lir.ece.calculate_ece(lrs, y, np.array([.5, .5]))

            self.assertEqual(context.exception.args[0], msg)

if __name__ == '__main__':
    unittest.main()
