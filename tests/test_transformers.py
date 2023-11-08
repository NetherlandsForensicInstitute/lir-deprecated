#!/usr/bin/env python3

import numpy as np
import unittest
import warnings

from scipy.stats import rankdata

from lir.transformers import InstancePairing, PercentileRankTransformer

warnings.simplefilter("error")


class TestPercentileRankTransformer(unittest.TestCase):
    def test_fit_transform(self):
        """When X itself is transformed, it should return its own ranks"""
        X = np.array([[0.1, 0.4, 0.5],
                      [0.2, 0.5, 0.55],
                      [0.15, 0.51, 0.55],
                      [0.18, 0.45, 0.56]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(X)
        self.assertSequenceEqual(ranks.tolist(),
                                 (rankdata(X, method='max', axis=0)/len(X)).tolist(),
                                 'Ranking X and PercentileRankTransformer.transform(X)'
                                 ' should give the same results')

    def test_extrapolation(self):
        """Values smaller than the lowest value should map to 0,
        values larger than the highest value should map to 1"""
        X = np.array([[0.1, 0.2, 0.3],
                      [0.2, 0.2, 0.4],
                      [0.3, 0.2, 0.5]])
        Z = np.array([[0.0, 0.1, 0.2],
                      [1.0, 1.0, 1.0]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        self.assertSequenceEqual(ranks.tolist(), [[0, 0, 0], [1, 1, 1]],
                                 'Elements smaller than the lowest value should'
                                 'map to 0, larger than the highest value to 1')

    def test_interpolation(self):
        """Values inbetween values of X result in interpolated ranking,
        with linear interpolation."""
        X = np.array([[0, 0, 0],
                      [1, 1, 1]])
        Z = np.array([[0.1, 0.3, 0.5]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        # Ranks are interpolated between 0.5 (rank of 0) and 1 (rank of 1).
        # We expect a linear interpolation.
        expected_ranks = 0.5 + np.array([[0.1, 0.3, 0.5]])*0.5
        self.assertSequenceEqual(ranks.tolist(), expected_ranks.tolist(),
                                 'Interpolation failed.')

    def test_ties(self):
        """Ties are given the maximum value (the maximum of the ranks that would
        have been assigned to all the tied values is assigned to each value)."""
        X = np.array([[0.1], [0.1], [0.1], [0.1], [0.3]])
        Z = np.array([[0.1]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        self.assertEqual(ranks, 0.8, "Ties should be given the maximum value")

    def test_constant_feature(self):
        """If a feature is a constant value c, the rank should be 0 for x < c
        and 1 for x >= c."""
        X = np.array([[1], [1], [1], [1]])
        Z = np.array([[0], [1], [2]])
        rank_transformer = PercentileRankTransformer()
        rank_transformer.fit(X)
        ranks = rank_transformer.transform(Z)
        self.assertEqual(ranks.tolist(), [[0], [1], [1]],
                         "If a feature is a constant value, "
                         "interpolation should still work")


class TestPairing(unittest.TestCase):
    X = np.arange(30).reshape(10, 3)
    y = np.concatenate([np.arange(5), np.arange(5)])

    def test_pairing(self):
        pairing = InstancePairing()
        X_pairs, y_pairs = pairing.transform(self.X, self.y)

        self.assertEqual(np.sum(y_pairs == 1), 5, 'number of same source pairs')
        self.assertEqual(np.sum(y_pairs == 0), 2*(8+6+4+2), 'number of different source pairs')

        pairing = InstancePairing(ratio_limit=1)
        X_pairs, y_pairs = pairing.transform(self.X, self.y)

        self.assertEqual(np.sum(y_pairs == 1), 5, 'number of same source pairs')
        self.assertEqual(np.sum(y_pairs == 0), 5, 'number of different source pairs')

        self.assertTrue(np.all(pairing.pairing[:, 0] != pairing.pairing[:, 1]), 'identity in pairs')

    def test_pairing_ratio(self):
        # test ratio
        ratio_limit = 7
        pairing_ratio = InstancePairing(ratio_limit=ratio_limit)
        X_pairs, y_pairs = pairing_ratio.transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertEqual(ratio, ratio_limit, 'ratio ss ds pairs not correct')

        # if ratio_limit exceeds highest possible ratio, all ds pairs are selected
        ratio_limit = 1_000
        pairing_ratio_max = InstancePairing(ratio_limit=ratio_limit)
        X_pairs, y_pairs = pairing_ratio_max.transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertLess(ratio, ratio_limit,
                        'realised ratio should be less or equal than ratio_limit')
        self.assertEqual(np.sum(y_pairs == 0), 2*(8+6+4+2),
                         'all different source pairs should be selected')

        # test ratio with same_source_limit
        ratio_limit = 5
        same_source_limit = 3
        pairing_ratio_ss_lim = InstancePairing(ratio_limit=ratio_limit,
                                            same_source_limit=same_source_limit)
        X_pairs, y_pairs = pairing_ratio_ss_lim.transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertEqual(ratio, ratio_limit, 'ratio ss ds pairs not correct')
        self.assertEqual(same_source_limit, np.sum(y_pairs == 1), 'ss pairs limit')

        # test ratio with different_source_limit
        ratio_limit = 5
        different_source_limit = 20
        pairing_ratio_ds_lim = InstancePairing(ratio_limit=ratio_limit,
                                               different_source_limit=different_source_limit)
        X_pairs, y_pairs = pairing_ratio_ds_lim.transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertLessEqual(ratio, ratio_limit, 'ratio ss ds pairs not correct')
        self.assertLessEqual(np.sum(y_pairs == 0), different_source_limit,
                         'ds pairs limit')

        # test ratio with same_source_limit and different_source_limit
        max_ratio = 5
        same_source_limit = 9
        different_source_limit = 30
        pairing_ratio_ds_lim = InstancePairing(ratio_limit=ratio_limit,
                                               same_source_limit=same_source_limit,
                                               different_source_limit=different_source_limit)
        X_pairs, y_pairs = pairing_ratio_ds_lim.transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertLessEqual(ratio, ratio_limit, 'ratio_limit ss and ds pairs exceeded')
        self.assertLessEqual(np.sum(y_pairs == 1), same_source_limit,
                             'ss pairs limit')
        self.assertLessEqual(np.sum(y_pairs == 0), different_source_limit,
                             'ds pairs limit')

    def test_pairing_seed(self):
        pairing_seed_1 = InstancePairing(ratio_limit=1, seed=123)
        X_pairs_1, y_pairs_1 = pairing_seed_1.transform(self.X, self.y)

        pairing_seed_2 = InstancePairing(ratio_limit=1, seed=123)
        X_pairs_2, y_pairs_2 = pairing_seed_2.transform(self.X, self.y)

        pairing_seed_3 = InstancePairing(ratio_limit=1, seed=456)
        X_pairs_3, y_pairs_3 = pairing_seed_3.transform(self.X, self.y)

        self.assertTrue(np.all(X_pairs_1 == X_pairs_2), 'same seed, same X pairs')
        self.assertTrue(np.any(X_pairs_1 != X_pairs_3), 'different seed, different X pairs')


if __name__ == '__main__':
    unittest.main()
