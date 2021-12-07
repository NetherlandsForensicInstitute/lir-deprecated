import numpy as np
import unittest
import warnings

from context import lir
assert lir  # so import optimizer doesn't remove the line above

from lir.calibration import IsotonicCalibrator
from lir.calibration import KDECalibrator, KDECalibratorInProbabilityDomain
from lir.calibration import GaussianCalibrator, GaussianCalibratorInProbabilityDomain
from lir.calibration import LogitCalibrator, LogitCalibratorInProbabilityDomain


from lir.util import Xn_to_Xy, Xy_to_Xn, to_probability, to_log_odds
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


# the X-data for TestKDECalibrator,  TestGaussianCalibrator, TestLogitCalibrator comes from random draws of perfectly calibrated LLR-distributions with mu_s = 6. For larger datasets it is confirmed that the calibration function approaches the line Y = X. The data under H1 and H0 are the 1:10 elements..
class TestKDECalibrator(unittest.TestCase):
    score_class0 = [9.10734621e-02, 1.37045394e-06, 7.09420701e-07, 5.71489514e-07, 2.44360004e-02, 5.53264987e-02,
                    6.40338659e-04, 8.22553310e-09, 2.57792725e-06]
    score_class1 = [2.42776744e+05, 5.35255527e+03, 1.50355963e+03, 1.08776892e+03, 2.19083530e+01, 7.13508826e+02,
                    2.23486401e+03, 5.52239060e+03, 1.12077833e+07]

    def test_log_odds_version(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = to_probability(X)
        desired = [3.59562799e-02, 1.75942116e-11, 2.59633540e-12, 1.36799721e-12, 8.15673411e-03, 2.10030624e-02, 3.70456430e-05, 1.40710861e-18, 1.04459592e-10, 3.14589737e+03, 2.59568527e+02, 1.08519904e+02, 8.56459139e+01, 3.81243702e+00, 6.23873841e+01, 1.43844114e+02, 2.64913149e+02, 1.49097168e+05]
        calibrator = KDECalibrator()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired)

    def test_prob_version(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = to_probability(X)
        X = to_log_odds(X)
        # desired is same as directly above
        desired = [3.59562799e-02, 1.75942116e-11, 2.59633540e-12, 1.36799721e-12, 8.15673411e-03, 2.10030624e-02, 3.70456430e-05, 1.40710861e-18, 1.04459592e-10, 3.14589737e+03, 2.59568527e+02, 1.08519904e+02, 8.56459139e+01, 3.81243702e+00, 6.23873841e+01, 1.43844114e+02, 2.64913149e+02, 1.49097168e+05]
        calibrator = KDECalibratorInProbabilityDomain()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired)

    def test_log_odds_version_on_extreme_values(self):
        X = np.array([8.34714300e-002, 1.37045206e-006, 7.09420198e-007, 5.71489187e-007, 2.38531254e-002, 5.24259542e-002, 6.39928887e-004, 8.22553304e-009, 2.57792061e-006, 0.00000000e+000, 9.88131292e-324, 0.00000000e+000, 9.99995881e-001, 9.99813208e-001, 9.99335354e-001, 9.99081531e-001, 9.56347800e-001, 9.98600437e-001, 9.99552746e-001, 9.99818952e-001, 9.99999911e-001, 1.00000000e+000, 1 - np.float_power(10, -16), 1.00000000e+000])
        y = np.concatenate((np.zeros(12), np.ones(12)))
        desired = [6.148510640582358, 0.10548096579142373, 0.07571171879632102, 0.06774859414831141, 4.408883097248305, 5.446103603204983, 1.4258427450086562, 0.006102474459494191, 0.14360453961912525, 0.0, 0.0, 0.0, 17.786943105214274, 21.248067409078676, 21.10676921763807, 20.955468109356307, 16.029054988277238, 20.689727349181517, 21.22851434841379, 21.24246276550688, 11.31919250180751, math.inf, 2.846712755553574, math.inf]
        calibrator = KDECalibrator()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired)


class TestGaussianCalibrator(unittest.TestCase):
    score_class0 = [9.10734621e-02, 1.37045394e-06, 7.09420701e-07, 5.71489514e-07, 2.44360004e-02, 5.53264987e-02,
                    6.40338659e-04, 8.22553310e-09, 2.57792725e-06]
    score_class1 = [2.42776744e+05, 5.35255527e+03, 1.50355963e+03, 1.08776892e+03, 2.19083530e+01, 7.13508826e+02,
                    2.23486401e+03, 5.52239060e+03, 1.12077833e+07]

    def test_log_odds_version(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = to_probability(X)
        desired = [3.06533372e-02, 5.92376598e-09, 1.96126088e-09, 1.35801711e-09, 6.71884519e-03, 1.74203890e-02, 6.47951643e-05, 6.33299969e-13, 1.67740156e-08, 2.38575352e+03, 3.62962152e+02, 1.65709775e+02, 1.33992382e+02, 6.90693424e+00, 1.00822553e+02, 2.13452386e+02, 3.69664971e+02, 7.74656845e+03]
        calibrator = GaussianCalibrator()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired)

    def test_prob_version(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = to_probability(X)
        X = to_log_odds(X)
        #desired is same as directly above
        desired = [3.06533372e-02, 5.92376598e-09, 1.96126088e-09, 1.35801711e-09, 6.71884519e-03, 1.74203890e-02, 6.47951643e-05, 6.33299969e-13, 1.67740156e-08, 2.38575352e+03, 3.62962152e+02, 1.65709775e+02, 1.33992382e+02, 6.90693424e+00, 1.00822553e+02, 2.13452386e+02, 3.69664971e+02, 7.74656845e+03]
        calibrator = GaussianCalibratorInProbabilityDomain()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired)

    def test_log_odds_version_on_extreme_values(self):
        X = np.array([8.34714300e-002, 1.37045206e-006, 7.09420198e-007, 5.71489187e-007, 2.38531254e-002, 5.24259542e-002, 6.39928887e-004, 8.22553304e-009, 2.57792061e-006, 0.00000000e+000, 9.88131292e-324, 0.00000000e+000, 9.99995881e-001, 9.99813208e-001, 9.99335354e-001, 9.99081531e-001, 9.56347800e-001, 9.98600437e-001, 9.99552746e-001, 9.99818952e-001, 9.99999911e-001, 1.00000000e+000, 1 - np.float_power(10, -16), 1.00000000e+000])
        y = np.concatenate((np.zeros(12), np.ones(12)))
        desired = [10.32769696, 0.74618471, 0.60926507, 0.56937812, 8.17855707, 9.47736296, 3.84324551, 0.13454357, 0.90195449, 0., 0., 0., 33.61358218, 31.95734672, 30.21933616, 29.69865022, 21.78925988, 28.97856863, 30.81599664, 31.99346455, 29.61269015, np.inf, 0.7317451, np.inf]
        calibrator = GaussianCalibrator()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired, rtol=1e-3)


class TestLogitCalibrator(unittest.TestCase):
    score_class0 = [9.10734621e-02, 1.37045394e-06, 7.09420701e-07, 5.71489514e-07, 2.44360004e-02, 5.53264987e-02,
                    6.40338659e-04, 8.22553310e-09, 2.57792725e-06]
    score_class1 = [2.42776744e+05, 5.35255527e+03, 1.50355963e+03, 1.08776892e+03, 2.19083530e+01, 7.13508826e+02,
                    2.23486401e+03, 5.52239060e+03, 1.12077833e+07]

    def test_log_odds_version(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = to_probability(X)
        desired = [1.79732352e-01, 4.16251897e-04, 2.90464504e-04, 2.58097514e-04, 8.75801433e-02, 1.36880766e-01, 1.19709662e-02, 2.54277585e-05, 5.87902757e-04, 5.83462439e+02, 7.25669320e+01, 3.62580218e+01, 3.03795948e+01, 3.59619089e+00, 2.41271821e+01, 4.50261471e+01, 7.38162334e+01, 4.73670703e+03]
        calibrator = LogitCalibrator()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired)

    def test_prob_version(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = to_probability(X)
        X = to_log_odds(X)
        # desired is same as directly above
        desired = [1.79732352e-01, 4.16251897e-04, 2.90464504e-04, 2.58097514e-04, 8.75801433e-02, 1.36880766e-01, 1.19709662e-02, 2.54277585e-05, 5.87902757e-04, 5.83462439e+02, 7.25669320e+01, 3.62580218e+01, 3.03795948e+01, 3.59619089e+00, 2.41271821e+01, 4.50261471e+01, 7.38162334e+01, 4.73670703e+03]
        calibrator = LogitCalibratorInProbabilityDomain()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired)

    def test_log_odds_version_on_extreme_values(self):
        X = np.array([8.34714300e-002, 1.37045206e-006, 7.09420198e-007, 5.71489187e-007, 2.38531254e-002, 5.24259542e-002, 6.39928887e-004, 8.22553304e-009, 2.57792061e-006, 0.00000000e+000, 9.88131292e-324, 0.00000000e+000, 9.99995881e-001, 9.99813208e-001, 9.99335354e-001, 9.99081531e-001, 9.56347800e-001, 9.98600437e-001, 9.99552746e-001, 9.99818952e-001, 9.99999911e-001, 1.00000000e+000, 1 - np.float_power(10, -16), 1.00000000e+000])
        y = np.concatenate((np.zeros(12), np.ones(12)))
        desired = [1.79732355e-001, 4.16251962e-004, 2.90464552e-004, 2.58097558e-004, 8.75801462e-002, 1.36880769e-001, 1.19709672e-002, 2.54277642e-005, 5.87902845e-004, 0.00000000e+000, 2.07535143e-177, 0.00000000e+000, 5.83462338e+002, 7.25669230e+001, 3.62580179e+001, 3.03795916e+001, 3.59619069e+000, 2.41271798e+001, 4.50261420e+001, 7.38162243e+001, 4.73670598e+003, np.inf, 3.48058077e+008, np.inf]
        calibrator = LogitCalibrator()
        calibrator.fit(X, y)
        lrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(lrs_cal, desired, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()


