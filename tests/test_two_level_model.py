import unittest
import os
from typing import Iterable

import numpy as np

from lir.classifiers import TwoLevelModelNormalKDE


dirname = os.path.dirname(__file__)
input_path = os.path.join(dirname, 'resources/two_level_model/input')
output_path = os.path.join(dirname, 'resources/two_level_model/R_output')

data_train = np.loadtxt(os.path.join(input_path, 'train_data.csv'), delimiter=",", dtype="float", skiprows=1, usecols=range(1, 12))
data_ref = np.loadtxt(os.path.join(input_path, 'reference_data.csv'), delimiter=",", dtype="float", skiprows=1, usecols=range(11))
data_tr = np.loadtxt(os.path.join(input_path, 'trace_data.csv'), delimiter=",", dtype="float", skiprows=1, usecols=range(1, 12))

mean_cov_within_R = np.loadtxt(os.path.join(output_path, 'MSwithin.csv'), delimiter=",", dtype="float", skiprows=1)
means_train_R = np.loadtxt(os.path.join(output_path, 'means_z.csv'), delimiter=",", dtype="float", skiprows=1)
kernel_bandwidth_sq_R = np.loadtxt(os.path.join(output_path, 'h2.csv'), delimiter=",", dtype="float", skiprows=1)
between_covars_R = np.loadtxt(os.path.join(output_path, 'T0.csv'), delimiter=",", dtype="float", skiprows=1)
covars_trace_R = np.loadtxt(os.path.join(output_path, 'U_h0.csv'), delimiter=",", dtype="float", skiprows=1)
covars_trace_update_R = np.loadtxt(os.path.join(output_path, 'U_hn.csv'), delimiter=",", dtype="float", skiprows=1)
covars_ref_R = np.loadtxt(os.path.join(output_path, 'U_hx.csv'), delimiter=",", dtype="float", skiprows=1)
updated_ref_mean_R = np.loadtxt(os.path.join(output_path, 'mu_h.csv'), delimiter=",", dtype="float", skiprows=1)
ln_num1_R = np.loadtxt(os.path.join(output_path, 'ln_num1.csv'), delimiter=",", dtype="float", skiprows=1)
ln_den_left_R = np.loadtxt(os.path.join(output_path, 'ln_num2.csv'), delimiter=",", dtype="float", skiprows=1)
ln_den_right_R = np.loadtxt(os.path.join(output_path, 'ln_den.csv'), delimiter=",", dtype="float", skiprows=1)
log10_LR_R = np.loadtxt(os.path.join(output_path, 'log10_MLRs.csv'), delimiter=",", dtype="float", skiprows=1)


def construct_3d_input(samples: Iterable[np.array]) -> np.ndarray:

    max_repetitions = max(s.shape[0] for s in samples)

    full_samples = []
    for sample in samples:
        full_sample = np.full((max_repetitions, samples[0].shape[1]), np.nan)
        full_sample.flat[:len(sample.flatten())] = sample.flatten()
        full_samples.append(full_sample)

    return np.array(full_samples)


class TestTwoLevelModelNormalKDEFit(unittest.TestCase):
    two_level_model = TwoLevelModelNormalKDE()

    def test_n_sources(self):
        n_sources = TwoLevelModelNormalKDE._get_n_sources(data_train[:, 0])
        np.testing.assert_equal(n_sources, 659)

    def test_n_features(self):
        n_features = TwoLevelModelNormalKDE._get_n_features(data_train[:, 1:], feature_ix=1)
        np.testing.assert_equal(n_features, 10)

    def test_mean_covariance_within(self):
        mean_cov_within_P = TwoLevelModelNormalKDE._get_mean_covariance_within(data_train[:, 1:],
                                                                             data_train[:, 0])
        np.testing.assert_almost_equal(mean_cov_within_P, mean_cov_within_R, decimal=17)

    def test_means_train(self):
        means_train_R_T = means_train_R.transpose()
        means_train_P = self.two_level_model._get_means_per_source(data_train[:, 1:], data_train[:, 0])
        np.testing.assert_almost_equal(means_train_P, means_train_R_T, decimal=14)

    def test_kernel_bandwidth_sq(self):
        kernel_bandwidth_sq_P = TwoLevelModelNormalKDE._get_kernel_bandwidth_squared(659, 10)
        np.testing.assert_almost_equal(kernel_bandwidth_sq_P, kernel_bandwidth_sq_R, decimal=16)

    def test_between_covars(self):
        between_covars_P = TwoLevelModelNormalKDE._get_between_covariance(data_train[:, 1:], data_train[:, 0], mean_cov_within_R)
        np.testing.assert_almost_equal(between_covars_P, between_covars_R, decimal=15)


class TestTwoLevelModelNormalKDEPredict(unittest.TestCase):
    two_level_model = TwoLevelModelNormalKDE()

    y = data_train[:, 0]

    # set or load output from fit function
    two_level_model.n_sources = 659
    two_level_model.n_features_train = 10
    two_level_model.mean_within_covars = mean_cov_within_R
    two_level_model.kernel_bandwidth_sq = kernel_bandwidth_sq_R
    two_level_model.between_covars = between_covars_R
    means_per_source_T = means_train_R
    two_level_model.means_per_source = means_per_source_T.transpose()

    # set 'model_fitted' to True
    two_level_model.model_fitted = True

    def test_U_h0(self):
        covars_trace_P = self.two_level_model._predict_covariances_trace_ref(data_train[[0, 1], 1:], data_ref)[0]
        np.testing.assert_almost_equal(covars_trace_P, covars_trace_R, decimal=15)

    def test_U_hn(self):
        covars_trace_update_P = self.two_level_model._predict_covariances_trace_ref(data_train[[0, 1], 1:], data_ref)[1]
        np.testing.assert_almost_equal(covars_trace_update_P, covars_trace_update_R, decimal=15)

    def test_U_hx(self):
        covars_ref_P = self.two_level_model._predict_covariances_trace_ref(data_train[[0, 1], 1:], data_ref)[2]
        np.testing.assert_almost_equal(covars_ref_P, covars_ref_R, decimal=15)

    def test_U_h0_inv(self):
        covars_trace_inv_P = self.two_level_model._predict_covariances_trace_ref(data_train[[0, 1], 1:], data_ref)[3]
        np.testing.assert_almost_equal(np.linalg.inv(covars_trace_inv_P), covars_trace_R, decimal=15)

    def test_U_hn_inv(self):
        covars_trace_update_inv_P = self.two_level_model._predict_covariances_trace_ref(data_train[[0, 1], 1:],
                                                                                        data_ref)[4]
        np.testing.assert_almost_equal(np.linalg.inv(covars_trace_update_inv_P), covars_trace_update_R, decimal=15)

    def test_U_hx_inv(self):
        covars_ref_inv_P = self.two_level_model._predict_covariances_trace_ref(data_train[[0, 1], 1:], data_ref)[5]
        np.testing.assert_almost_equal(np.linalg.inv(covars_ref_inv_P), covars_ref_R, decimal=15)

    def test_mu_h(self):
        covars_ref_inv = np.linalg.inv(covars_ref_R)

        updated_ref_mean_P = self.two_level_model._predict_updated_ref_mean(data_ref[:, 1:], covars_ref_inv)
        np.testing.assert_almost_equal(updated_ref_mean_P.transpose(), updated_ref_mean_R, decimal=13)

    def test_ln_num(self):
        # load precalculated parameters that have already been predicted and are necessary input for current test
        covars_ref_inv = np.linalg.inv(covars_ref_R)
        covars_trace_update_inv = np.linalg.inv(covars_trace_update_R)
        updated_ref_mean_T = updated_ref_mean_R.transpose()

        # calculate test object and compare
        ln_num_P = self.two_level_model._predict_ln_num(data_tr[[0, 1], 1:], data_ref[:, 1:], covars_ref_inv,
                                                        covars_trace_update_inv, updated_ref_mean_T)
        np.testing.assert_almost_equal(ln_num_P, ln_num1_R, decimal=14)

    def test_ln_den_left(self):
        # load precalculated parameters that have already been predicted and are necessary input for current test
        covars_ref_inv = np.linalg.inv(covars_ref_R)
        # calculate test object and compare
        ln_den_left_P = self.two_level_model._predict_ln_den_term(data_ref[:, 1:], covars_ref_inv)
        np.testing.assert_almost_equal(ln_den_left_P, ln_den_left_R, decimal=14)

    def test_ln_den_right(self):
        # load precalculated parameters that have already been predicted and are necessary input for current test
        covars_trace_inv = np.linalg.inv(covars_trace_R)
        # calculate test object and compare
        ln_den_right_P = self.two_level_model._predict_ln_den_term(data_tr[[0, 1], 1:], covars_trace_inv)
        np.testing.assert_almost_equal(ln_den_right_P, ln_den_right_R, decimal=14)

    def test_log10_LR_from_formula_Bolck(self):
        log10_LR = log10_LR_R[0]
        log10_LR_P = self.two_level_model._predict_log10_LR_from_formula_Bolck(covars_trace_R, covars_trace_update_R,
                                                                               ln_num1_R, ln_den_left_R, ln_den_right_R)
        np.testing.assert_almost_equal(log10_LR_P, log10_LR, decimal=13)

    def test_predict_log10_LR_score(self):

        data_tr_samples = [data_tr[data_tr[:, 0] == label, 1:] for label in np.unique(data_tr[:, 0])]
        data_tr_reshaped = construct_3d_input(data_tr_samples)

        data_ref_samples = [data_ref[:, 1:] for i in data_tr_samples]
        data_ref_reshaped = construct_3d_input(data_ref_samples)

        log10_LR_P = self.two_level_model._predict_log10_lr_score(data_tr_reshaped, data_ref_reshaped)

        # replace too negative log10_LR_P since log10_LR_R gives -Inf after -300
        log10_LR_P[log10_LR_P < -300] = -np.inf

        np.testing.assert_almost_equal(np.array(log10_LR_R), log10_LR_P, decimal=10)


class TestTwoLevelModelNormalKDEFitPredict(unittest.TestCase):
    two_level_model = TwoLevelModelNormalKDE()

    y = data_train[:, 0]

    def test_fit_and_predict_log10_LR_score(self):
        # load in ground truth LLRs and instantiate calculated LLRs list
        log10_LR = np.array(log10_LR_R)

        self.two_level_model.fit_on_unpaired_instances(data_train[:, 1:], self.y)

        data_tr_samples = [data_tr[data_tr[:, 0] == label, 1:] for label in np.unique(data_tr[:, 0])]
        data_tr_reshaped = construct_3d_input(data_tr_samples)

        data_ref_samples = [data_ref[:,1:] for i in data_tr_samples]
        data_ref_reshaped = construct_3d_input(data_ref_samples)

        log10_LR_P = self.two_level_model._predict_log10_lr_score(data_tr_reshaped, data_ref_reshaped)

        # replace too negative log10_LR_P since log10_LR_R gives -Inf after -300
        log10_LR_P[log10_LR_P < -300] = -np.inf

        np.testing.assert_almost_equal(log10_LR, log10_LR_P, decimal=10)

    def test_fit_and_transform(self):
        # load in ground truth LLRs and instantiate calculated LLRs list
        odds_R = np.array(10 ** log10_LR_R)

        self.two_level_model.fit_on_unpaired_instances(data_train[:, 1:], self.y)

        data_tr_samples = [data_tr[data_tr[:, 0] == label, 1:] for label in np.unique(data_tr[:, 0])]
        data_tr_reshaped = construct_3d_input(data_tr_samples)

        data_ref_samples = [data_ref[:, 1:] for i in data_tr_samples]
        data_ref_reshaped = construct_3d_input(data_ref_samples)

        LRs_P = self.two_level_model.transform(data_tr_reshaped, data_ref_reshaped)

        # create ground truth
        ground_truth = np.repeat(1.0, len(LRs_P))
        ground_truth[odds_R == 0] = float('nan')
        odds_R[odds_R == 0] = float('nan')

        np.testing.assert_almost_equal(LRs_P / odds_R, ground_truth, decimal=9)

    def test_fit_and_predict_proba(self):
        # instantiate calculated LLRs list
        odds_R = np.array(10 ** log10_LR_R)
        p1_R = odds_R / (1 + odds_R)
        p0_R = 1 - p1_R
        probs_R = np.transpose(np.array((p0_R, p1_R)))

        self.two_level_model.fit_on_unpaired_instances(data_train[:, 1:], self.y)

        data_tr_samples = [data_tr[data_tr[:, 0] == label, 1:] for label in np.unique(data_tr[:, 0])]
        data_tr_reshaped = construct_3d_input(data_tr_samples)

        data_ref_samples = [data_ref[:, 1:] for i in data_tr_samples]
        data_ref_reshaped = construct_3d_input(data_ref_samples)

        probs_P = self.two_level_model.predict_proba(data_tr_reshaped, data_ref_reshaped)

        np.testing.assert_almost_equal(probs_P, probs_R, decimal=16)


if __name__ == '__main__':
    unittest.main()
