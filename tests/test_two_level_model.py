import unittest
import os
import numpy as np

from lir.classifiers.two_level_model import TwoLevelModel


class TestTwoLevelModel_fit_functions(unittest.TestCase):

    two_level_model = TwoLevelModel()

    dirname = os.path.dirname(__file__)
    data_train = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/input/train_data.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))

    def test_n_sources(self):
        n_sources = self.two_level_model._fit_n_sources(self.data_train[:, 0])
        np.testing.assert_equal(n_sources, 659)

    def test_mean_covariance_within(self):
        mean_cov_within_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        mean_cov_within_P = self.two_level_model._fit_mean_covariance_within(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(mean_cov_within_P, mean_cov_within_R, decimal=17)

    def test_means_train(self):
        means_train_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/means_z.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        means_train_R = means_train_R.transpose()
        means_train_P = self.two_level_model._fit_means_per_source(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(means_train_P, means_train_R, decimal=14)

    def test_kernel_bandwidth_sq(self):
        kernel_bandwidth_sq_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/h2.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        self.two_level_model.n_sources = 659
        kernel_bandwidth_sq_P = self.two_level_model._fit_kernel_bandwidth_squared(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(kernel_bandwidth_sq_P, kernel_bandwidth_sq_R, decimal=16)

    def test_between_covars(self):
        between_covars_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/T0.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        mean_cov_within_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/MSwithin.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        self.two_level_model.mean_within_covars = mean_cov_within_R
        self.two_level_model.model_fitted = True
        between_covars_P = self.two_level_model._fit_between_covariance(self.data_train[:, 1:], self.data_train[:, 0])
        np.testing.assert_almost_equal(between_covars_P, between_covars_R, decimal=15)

class Test_TwoLevelModel_predict_functions(unittest.TestCase):
    two_level_model = TwoLevelModel()

    dirname = os.path.dirname(__file__)
    # load datasets
    data_train = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/input/train_data.csv'), delimiter=",",
                            dtype="float", skiprows=1,
                            usecols=range(1, 12))
    y = data_train[:,0]
    data_ref = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/input/reference_data.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 11))
    data_tr = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/input/trace_data.csv'), delimiter=",", dtype="float", skiprows=1,
                       usecols=range(1, 12))
    # set or load output from fit function
    two_level_model.n_sources = 659
    two_level_model.mean_within_covars = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/R_output/MSwithin.csv'),
                                   delimiter=","
                                   , dtype="float", skiprows=1)
    two_level_model.kernel_bandwidth_sq = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/R_output/h2.csv'),
                                       delimiter=","
                                       , dtype="float", skiprows=1)
    two_level_model.between_covars = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/R_output/T0.csv'), delimiter=","
                      , dtype="float", skiprows=1)
    means_per_source_T = np.loadtxt(os.path.join(dirname, 'resources/two_level_model/R_output/means_z.csv'),
                                    delimiter=","
                                    , dtype="float", skiprows=1)
    two_level_model.means_per_source = means_per_source_T.transpose()

    # set 'model_fitted' to True
    two_level_model.model_fitted = True

    def test_U_h0(self):
        covars_trace_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_h0.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        covars_trace_P = self.two_level_model._predict_covariances_trace_ref(self.data_train[[0, 1, 2], 1:], self.data_ref)[0]
        np.testing.assert_almost_equal(covars_trace_P, covars_trace_R, decimal=15)

    def test_U_hn(self):
        covars_trace_update_R  = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hn.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        covars_trace_update_P = self.two_level_model._predict_covariances_trace_ref(self.data_train[[0, 1, 2], 1:], self.data_ref)[1]
        np.testing.assert_almost_equal(covars_trace_update_P, covars_trace_update_R, decimal=15)

    def test_U_hx(self):
        covars_ref_R  = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hx.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        covars_ref_P = self.two_level_model._predict_covariances_trace_ref(self.data_train[[0, 1, 2], 1:], self.data_ref)[2]
        np.testing.assert_almost_equal(covars_ref_P, covars_ref_R, decimal=15)

    def test_U_h0_inv(self):
        covars_trace_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_h0.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        covars_trace_inv_P = \
        self.two_level_model._predict_covariances_trace_ref(self.data_train[[0, 1, 2], 1:], self.data_ref)[3]
        np.testing.assert_almost_equal(np.linalg.inv(covars_trace_inv_P), covars_trace_R, decimal=15)

    def test_U_hn_inv(self):
        covars_trace_update_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hn.csv'),
                                           delimiter=","
                                           , dtype="float", skiprows=1)
        covars_trace_update_inv_P = \
        self.two_level_model._predict_covariances_trace_ref(self.data_train[[0, 1, 2], 1:], self.data_ref)[4]
        np.testing.assert_almost_equal(np.linalg.inv(covars_trace_update_inv_P), covars_trace_update_R, decimal=15)

    def test_U_hx_inv(self):
        covars_ref_R  = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hx.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        covars_ref_inv_P = self.two_level_model._predict_covariances_trace_ref(self.data_train[[0, 1, 2], 1:], self.data_ref)[5]
        np.testing.assert_almost_equal(np.linalg.inv(covars_ref_inv_P), covars_ref_R, decimal=15)

    def test_mu_h(self):
        updated_ref_mean_R  = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/mu_h.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        # load precalculated parameters that have already been predicted and are necessarry input for current test
        covars_ref = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hx.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        covars_ref_inv = np.linalg.inv(covars_ref)

        # perform desired function
        updated_ref_mean_P = self.two_level_model._predict_updated_ref_mean(self.data_ref, covars_ref_inv)
        # test
        np.testing.assert_almost_equal(updated_ref_mean_P.transpose(), updated_ref_mean_R, decimal=13)

    def test_ln_num(self):
        ln_num1_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/ln_num1.csv'), delimiter=","
                            , dtype="float", skiprows=1)

        # load precalculated parameters that have already been predicted and are necessarry input for current test
        covars_ref = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hx.csv'), delimiter=","
                            , dtype="float", skiprows=1)
        covars_ref_inv = np.linalg.inv(covars_ref)
        covars_trace_update = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hn.csv'), delimiter=","
                            , dtype="float", skiprows=1)
        covars_trace_update_inv = np.linalg.inv(covars_trace_update)
        updated_ref_mean_T = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/mu_h.csv'), delimiter=","
                            , dtype="float", skiprows=1)
        updated_ref_mean = updated_ref_mean_T.transpose()
        # calculate test object and compare
        ln_num_P = self.two_level_model._predict_ln_num(self.data_tr[[0, 1, 2], 1:], self.data_ref, covars_ref_inv, covars_trace_update_inv, updated_ref_mean)
        np.testing.assert_almost_equal(ln_num_P, ln_num1_R, decimal=14)

    def test_ln_den_left(self):
        ln_den_left_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/ln_num2.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        # load precalculated parameters that have already been predicted and are necessarry input for current test
        covars_ref = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hx.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        covars_ref_inv = np.linalg.inv(covars_ref)
        # calculate test object and compare
        ln_den_left_P = self.two_level_model._predict_ln_den_term(self.data_ref, covars_ref_inv)
        np.testing.assert_almost_equal(ln_den_left_P, ln_den_left_R, decimal=14)

    def test_ln_den_right(self):
        ln_den_right_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/ln_den.csv'), delimiter=","
                               , dtype="float", skiprows=1)
        # load precalculated parameters that have already been predicted and are necessarry input for current test
        covars_trace = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_h0.csv'), delimiter=","
                                , dtype="float", skiprows=1)
        covars_trace_inv = np.linalg.inv(covars_trace)
        # calculate test object and compare
        ln_den_right_P = self.two_level_model._predict_ln_den_term(self.data_tr[[0, 1, 2], 1:], covars_trace_inv)
        np.testing.assert_almost_equal(ln_den_right_P, ln_den_right_R, decimal=14)

    def test_log10_LR_from_formula_Bolck(self):
        log10_LR_R = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/log10_MLRs.csv'), delimiter=","
                                , dtype="float", skiprows=1)[0]
        # load precalculated parameters that have already been predicted and are necessarry input for current test
        covars_trace = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_h0.csv'),
                                  delimiter=","
                                  , dtype="float", skiprows=1)
        covars_trace_update = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/U_hn.csv'),
                                  delimiter=","
                                  , dtype="float", skiprows=1)
        ln_num = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/ln_num1.csv'),
                                  delimiter=","
                                  , dtype="float", skiprows=1)
        ln_den_left = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/ln_num2.csv'),
                            delimiter=","
                            , dtype="float", skiprows=1)
        ln_den_right = np.loadtxt(os.path.join(self.dirname, 'resources/two_level_model/R_output/ln_den.csv'),
                            delimiter=","
                            , dtype="float", skiprows=1)
        # calculate test object and compare
        log10_LR_P = self.two_level_model._predict_log10_LR_from_formula_Bolck(covars_trace, covars_trace_update, ln_num, ln_den_left, ln_den_right)
        np.testing.assert_almost_equal(log10_LR_P, log10_LR_R, decimal=13)

if __name__ == '__main__':
    unittest.main()
