import csv
import os
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from lir.calibration import FourParameterLogisticCalibrator
from lir.metrics import devpav
from lir.util import to_probability, to_log_odds


def read_data(path):
    with open(path, 'r') as file:
        r = csv.reader(file)
        next(r)
        data = np.array([float(value) for _, value in r])
    return to_probability(np.array(data))


class TestFourParameterLogisticCalibrator(unittest.TestCase):
    dirname = os.path.dirname(__file__)
    X_diff = read_data(os.path.join(dirname, 'resources/LRsdifferentnormalLLRdistribmu_s=1N_ss=300.csv'))
    X_same = read_data(os.path.join(dirname, 'resources/LRssamenormalLLRdistribmu_s=1N_ss=300.csv'))

    def test_compare_to_logistic(self):
        X = np.concatenate([self.X_diff, self.X_same])
        X = to_log_odds(X)
        y = np.concatenate([np.zeros(len(self.X_diff)), np.ones(len(self.X_same))])
        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        logistic = LogisticRegression(penalty=None, class_weight='balanced')
        logistic.fit(X[:, None], y)
        logistic_coef = [logistic.coef_[0][0], logistic.intercept_[0]]
        np.testing.assert_allclose(four_pl_model.coef_, logistic_coef, rtol=1e-2)

    def test_pl_1_is_0(self):
        X_same = np.concatenate([self.X_same, [0]])
        X_diff = np.concatenate([self.X_diff, [0]])
        y = np.concatenate([np.zeros(len(X_diff)), np.ones(len(X_same))])
        X = np.concatenate([X_diff, X_same])
        X = to_log_odds(X)

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        odds = four_pl_model.transform(X)
        np.testing.assert_almost_equal(devpav(odds, y), 0.12029952948152635, decimal=5)

    def test_pl_0_is_1(self):
        X_same = np.concatenate([self.X_same, [1, 1 - 10 ** -10]])
        X_diff = np.concatenate([self.X_diff, [1, 1 - 10 ** -10]])
        y = np.concatenate([np.zeros(len(X_diff)), np.ones(len(X_same))])
        X = np.concatenate([X_diff, X_same])
        X = to_log_odds(X)

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        odds = four_pl_model.transform(X)
        np.testing.assert_almost_equal(devpav(odds, y), 0.15273304557837525, decimal=5)

    def test_pl_0_is_1_and_pl_1_is_0(self):
        X_same = np.concatenate([self.X_same, [0, 10 ** -10, 1, 1 - 10 ** -10]])
        X_diff = np.concatenate([self.X_diff, [0, 10 ** -10, 1, 1 - 10 ** -10]])
        y = np.concatenate([np.zeros(len(X_diff)), np.ones(len(X_same))])
        X = np.concatenate([X_diff, X_same])
        X = to_log_odds(X)

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        odds = four_pl_model.transform(X)
        np.testing.assert_almost_equal(devpav(odds, y), 0.10475112893952891, decimal=5)


if __name__ == '__main__':
    unittest.main()
