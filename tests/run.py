# read data
# calculate intermediate parameters
# fit TLM

import os
import numpy as np
from lir.classifiers import TLM_calc_MSwithin, TLM_calc_means, TLM_calc_h_sq, TLM_calc_T0, TLM_calc_U, TLM_calc_mu_h, \
    TLM_calc_ln_num, TLM_calc_ln_den_left, TLM_calc_ln_den_right

dirname = os.path.dirname(__file__)
dataZ = np.loadtxt(os.path.join(dirname, 'data/TLM/input/ZDATA.csv'), delimiter=",", dtype="float", skiprows=1, usecols=range(1,12))
MSwithin = TLM_calc_MSwithin(dataZ[:,1:], dataZ[:,0])
means_z = TLM_calc_means(dataZ[:,1:], dataZ[:,0])
h_sq = TLM_calc_h_sq(dataZ[:,1:], dataZ[:,0])
T0 = TLM_calc_T0(dataZ[:,1:], dataZ[:,0])

dataX = np.loadtxt(os.path.join(dirname, 'data/TLM/input/XDATA.csv'), delimiter=",", dtype="float", skiprows=1, usecols=range(1,11))
dataY = np.loadtxt(os.path.join(dirname, 'data/TLM/input/YDATA.csv'), delimiter=",", dtype="float", skiprows=1, usecols=range(1,12))

U_h0_inv, U_hx_inv, U_hn_inv = TLM_calc_U(dataY[[0, 1, 2], 1:], dataX, MSwithin, h_sq, T0)
mu_h = TLM_calc_mu_h(dataX, MSwithin, T0, h_sq, dataZ[:,1:], dataZ[:,0])
ln_num = TLM_calc_ln_num(dataY[[0, 1, 2], 1:], dataX, U_hx_inv, U_hn_inv, mu_h, dataZ[:,1:], dataZ[:,0])
ln_den_left = TLM_calc_ln_den_left(dataY[[0, 1, 2], 1:], U_hx_inv, dataZ[:, 1:], dataZ[:, 0])
ln_den_right = TLM_calc_ln_den_right(dataX, U_h0_inv, dataZ[:, 1:], dataZ[:, 0])
print("t")