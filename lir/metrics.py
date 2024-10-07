import collections
import warnings
from typing import List

import numpy as np

from .calibration import IsotonicCalibrator
from .util import Xn_to_Xy, Xy_to_Xn, to_probability, LR


LrStats = collections.namedtuple('LrStats',
                                 ['avg_log2lr', 'avg_log2lr_class0', 'avg_log2lr_class1', 'avg_p0_class0', 'avg_p1_class0',
                                  'avg_p0_class1', 'avg_p1_class1', 'cllr_class0', 'cllr_class1', 'cllr', 'lr_class0',
                                  'lr_class1', 'cllr_min', 'cllr_cal'])


def cllr(lrs, y, weights=(1, 1)):
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood
    ratios.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    Parameters
    ----------
    lrs : a numpy array of LRs
    y : a numpy array of labels (0 or 1)

    Returns
    -------
    cllr
        the log likelihood ratio cost
    """

    # ignore errors:
    #   divide -> ignore divide by zero
    #   over -> ignore scalar overflow
    with np.errstate(divide='ignore', over='ignore'):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        cllr0 = weights[0] * np.mean(np.log2(1 + lrs0)) if weights[0] > 0 else 0
        cllr1 = weights[1] * np.mean(np.log2(1 + 1 / lrs1)) if weights[1] > 0 else 0
        return (cllr0 + cllr1) / sum(weights)


def cllr_min(lrs, y, weights=(1, 1)):
    """
    Estimates the discriminative power from a collection of likelihood ratios.

    Parameters
    ----------
    lrs : a numpy array of LRs
    y : a numpy array of labels (0 or 1)

    Returns
    -------
    cllr_min
        the log likelihood ratio cost
    """
    cal = IsotonicCalibrator()
    lrmin = cal.fit_transform(to_probability(lrs), y)
    return cllr(lrmin, y, weights)


def _calcsurface(c1: (float, float), c2: (float, float)) -> float:
    """
    Helperfunction that calculates the desired surface for two xy-coordinates
    """
    # step 1: calculate intersection (xs, ys) of straight line through coordinates with identity line (if slope (a) = 1, there is no intersection and surface of this parrellogram is equal to deltaY * deltaX)


    x1, y1 = c1
    x2, y2 = c2
    a = (y2 - y1) / (x2 - x1)

    if a == 1:
        # dan xs equals +/- Infinite en is er there is no intersection with the identity line

        # the surface of the parallellogram is:
        surface = (x2 - x1) * np.abs(y1 - x1)

    elif (a < 0):
        raise ValueError(f"slope is negative; impossible for PAV-transform. Coordinates are {c1} and {c2}. Calculated slope is {a}")
    else:
        # than xs is finite:
        b = y1 - a * x1
        xs = b / (1 - a)
        # xs

        # step 2: check if intersection is located within line segment c1 and c2.
        if x1 < xs and x2 >= xs:
            # then intersection is within
            # (situation 1 of 2) if y1 <= x1 than surface is:
            if y1 <= x1:
                surface = 0.5 * (xs - y1) * (xs - x1) - 0.5 * (xs - x1) * (xs - x1) + 0.5 * (y2 - xs) * (x2 - xs) - 0.5 * (
                            x2 - xs) * (x2 - xs)
            else:
                # (situation 2 of 2) than y1 > x1, and surface is:
                surface = 0.5 * (xs - x1) ** 2 - 0.5 * (xs - y1) * (xs - x1) + 0.5 * (x2 - xs) ** 2 - 0.5 * (x2 - xs) * (
                            y2 - xs)
                # dit is the same as 0.5 * (xs - x1) * (xs - y1) - 0.5 * (xs - y1) * (xs - y1) + 0.5 * (y2 - xs) * (x2 - xs) - 0.5 * (y2 - xs) * (y2 - xs) + 0.5 * (y1 - x1) * (y1 - x1) + 0.5 * (x2 - y2) * (x2 -y2)
        else:  # then intersection is not within line segment
            # if (situation 1 of 4) y1 <= x1 AND y2 <= x1, and surface is
            if y1 <= x1 and y2 <= x1:
                surface = 0.5 * (y2 - y1) * (x2 - x1) + (x1 - y2) * (x2 - x1) + 0.5 * (x2 - x1) * (x2 - x1)
            elif y1 > x1:  # (situation 2 of 4) then y1 > x1, and surface is
                surface = 0.5 * (x2 - x1) * (x2 - x1) + (y1 - x2) * (x2 - x1) + 0.5 * (y2 - y1) * (x2 - x1)
            elif y1 <= x1 and y2 > x1:  # (situation 3 of 4). This should be the last possibility.
                surface = 0.5 * (y2 - y1) * (x2 - x1) - 0.5 * (y2 - x1) * (y2 - x1) + 0.5 * (x2 - y2) * (x2 - y2)
            else:
                # situation 4 of 4 : this situation should never appear. There is a fourth sibution as situation 3, but than above the identity line. However, this is impossible by definition of a PAV-transform (y2 > x1).
                raise ValueError(f"unexpected coordinate combination: ({x1}, {y1}) and ({x2}, {y2})")
    return surface


def _devpavcalculator(lrs, pav_lrs, y):
    """
    function that calculates davPAV for a PAVresult for SSLRs and DSLRs  een PAV transformatie de devPAV uitrekent
    Input: Lrs = np.array met LR-waarden. pav_lrs = np.array met uitkomst van PAV-transformatie op lrs. y = np.array met labels (1 voor H1 en 0 voor H2)
    Output: devPAV value

    """
    DSLRs, SSLRs = Xy_to_Xn(lrs, y)
    DSPAVLRs, SSPAVLRs = Xy_to_Xn(pav_lrs, y)
    PAVresult = np.concatenate([SSPAVLRs, DSPAVLRs])
    Xen = np.concatenate([SSLRs, DSLRs])

    # order coordinates based on x's then y's and filtering out identical datapoints
    data = np.unique(np.array([Xen, PAVresult]), axis=1)
    Xen = data[0, :]
    Yen = data[1, :]

    # pathological cases
    # first one of four: PAV-transform has a horizonal line to log(X) = -Inf as to log(X) = Inf
    if Yen[0] != 0 and Yen[-1] != np.inf and Xen[-1] == np.inf and Xen[-1] == np.inf:
        return np.inf

    # second of four: PAV-transform has a horizontal line to log(X) = -Inf
    if Yen[0] != 0 and Xen[0] == 0 and Yen[-1] == np.inf:
        return np.inf

    # third of four: PAV-transform has a horizontal line to log(X) = Inf
    if Yen[0] == 0 and Yen[-1] != np.inf and Xen[-1] == np.inf:
        return np.inf

    # forth of four: PAV-transform has one vertical line from log(Y) = -Inf to log(Y) = Inf
    wh = (Yen == 0) | (Yen == np.inf)
    if np.sum(wh) == len(Yen):
        return np.nan

    else:
        # then it is not a  pathological case with weird X-values and devPAV can be calculated

        # filtering out -Inf or 0 Y's
        wh = (Yen > 0) & (Yen < np.inf)
        Xen = np.log10(Xen[wh])
        Yen = np.log10(Yen[wh])
        # create an empty list with size (len(Xen))
        devPAVs = [None] * len(Xen)
        # sanity check
        if len(Xen) == 0:
            return np.nan
        elif len(Xen) == 1:
            return abs(Xen - Yen)
        # than calculate devPAV
        else:
            deltaX = Xen[-1] - Xen[0]
            surface = 0
            for i in range(1, (len(Xen))):
                surface = surface + _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
                devPAVs[i - 1] = _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
            # return(list(surface/a, PAVresult, Xen, Yen, devPAVs))
            return surface / deltaX


def devpav(lrs: np.ndarray, y: np.ndarray) -> float:
    """
    calculates devPAV for LR data under H1 and H2.
    """
    if sum(y) == len(y) or sum(y) == 0:
        raise ValueError('devpav: illegal input: at least one value is required for each class')
    cal = IsotonicCalibrator()
    pavlrs = cal.fit_transform(lrs, y)
    return _devpavcalculator(lrs, pavlrs, y)


def calculate_lr_statistics(lr_class0: List[LR], lr_class1: List[LR]) -> LrStats:
    """
    Calculates various statistics for a collection of likelihood ratios.

    Parameters
    ----------
    lr_class0 : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from class 0.
    lr_class1 : list of float
        Likelihood ratios which are calculated for measurements which are
        sampled from class 1.

    Returns
    -------
    LrStats
        Likelihood ratio statistics.
    """
    assert len(lr_class0) > 0
    assert len(lr_class1) > 0

    def avg(*args):
        return sum(args) / len(args)

    if type(lr_class0[0]) == LR:
        avg_p0_class0 = avg(*[lr.p0 for lr in lr_class0])
        avg_p1_class0 = avg(*[lr.p1 for lr in lr_class0])
        avg_p0_class1 = avg(*[lr.p0 for lr in lr_class1])
        avg_p1_class1 = avg(*[lr.p1 for lr in lr_class1])
        lr_class0 = np.array([lr.lr for lr in lr_class0])
        lr_class1 = np.array([lr.lr for lr in lr_class1])
    else:
        if type(lr_class0) == list:
            lr_class0 = np.array(lr_class0)
            lr_class1 = np.array(lr_class1)

        avg_p0_class0 = None
        avg_p1_class0 = None
        avg_p0_class1 = None
        avg_p1_class1 = None

    with warnings.catch_warnings():
        try:
            avg_log2lr_class0 = np.mean(np.log2(1 / lr_class0))
            avg_log2lr_class1 = np.mean(np.log2(lr_class1))
            avg_log2lr = avg(avg_log2lr_class0, avg_log2lr_class1)
        except RuntimeWarning:
            # possibly illegal LRs such as 0 or inf
            avg_log2lr_class0 = np.nan
            avg_log2lr_class1 = np.nan
            avg_log2lr = np.nan

    lrs, y = Xn_to_Xy(lr_class0, lr_class1)
    cllr_class0 = cllr(lrs, y, weights=(1, 0))
    cllr_class1 = cllr(lrs, y, weights=(0, 1))
    cllr_ = .5 * (cllr_class0 + cllr_class1)

    cllrmin_class0 = cllr_min(lrs, y, weights=(1, 0))
    cllrmin_class1 = cllr_min(lrs, y, weights=(0, 1))
    cllrmin = .5 * (cllrmin_class0 + cllrmin_class1)

    return LrStats(avg_log2lr, avg_log2lr_class0, avg_log2lr_class1,
                   avg_p0_class0, avg_p1_class0, avg_p0_class1, avg_p1_class1,
                   cllr_class0, cllr_class1, cllr_, lr_class0, lr_class1,
                   cllrmin, cllr_ - cllrmin)
