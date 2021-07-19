import collections
import warnings

import numpy as np

from .calibration import IsotonicCalibrator
from .util import Xn_to_Xy, Xy_to_Xn, to_probability, LR

LrStats = collections.namedtuple('LrStats',
                                 ['avg_llr', 'avg_llr_class0', 'avg_llr_class1', 'avg_p0_class0', 'avg_p1_class0',
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


def devpav_estimated(lrs, y, resolution=1000):
    """
    Estimate devPAV, a metric for calibration.

    devPAV is the cumulative deviation of the PAV transformation from
    the identity line. It is calculated in the LR range where misleading LRs
    occur.

    See also: P. Vergeer, Measuring calibration of likelihood ratio systems: a
    comparison of four systems, including a new metric devPAV, to appear

    This implementation estimates devPAV by calculating the average deviation
    for a large number of LRs.

    Parameters
    ----------
    lrs : a numpy array of LRs
    y : a numpy array of labels (0 or 1)
    resolution : the number of measurements in the range of misleading evidence; a higher value yields a more accurate estimation

    Returns
    -------
    devPAV
        an estimation of devPAV
    """
    lrs0, lrs1 = Xy_to_Xn(lrs, y)
    if len(lrs0) == 0 or len(lrs1) == 0:
        raise ValueError('devpav: illegal input: at least one value is required for each class')

    # find misleading LR extremes
    first_misleading = np.min(lrs1)
    last_misleading = np.max(lrs0)
    if first_misleading > last_misleading:  # test for perfect discrimination
        return 0

    if np.isinf(first_misleading) or np.isinf(last_misleading):  # test for infinitely misleading LRs
        return np.inf

    # calibrate on the input LRs
    cal = IsotonicCalibrator()
    cal.fit_transform(to_probability(lrs), y)

    # take `resolution` points evenly divided along the range of misleading LRs
    xlr = np.exp(np.linspace(np.log(first_misleading), np.log(last_misleading), resolution))
    pavlr = cal.transform(to_probability(xlr))

    devlr = np.absolute(np.log10(xlr) - np.log10(pavlr))
    return (np.sum(devlr) / resolution) * (np.log10(last_misleading) - np.log10(first_misleading))


def calcsurface_f(c1, c2):
    """
    Helperfunction that calculates the desired surface for two xy-coordinates
    """
    # step 1: calculate intersection (xs, ys) of straight line through coordinates with identity line (if slope (a) = 1, there is no intersection and surface of this parrellogram is equal to deltaY * deltaX)


    x1, y1 = c1
    x2, y2 = c2
    a = (y2 - y1) / (x2 - x1)

    if a == 1:
        # dan xs equals +/- Infinite en is er there is no intersection with the identity line
        # since condition 1 holds the product below is always positive
        surface = (y2 - y1) * (x2 - x1)
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
    DSLRs, SSLRs = Xy_to_Xn(lrs,y)
    DSPAVLRs, SSPAVLRs = Xy_to_Xn(pav_lrs, y)
    PAVresult = np.concatenate([SSPAVLRs, DSPAVLRs])
    Xen = np.concatenate([SSLRs, DSLRs])

    # order coordinates based on x's then y's and filtering out identical datapoints
    data = np.unique(np.array([Xen, PAVresult]), axis=1)
    Xen = data[0,:]
    Yen = data[1,:]
    # pathological cases
    # check if min(Xen) = 0 or max(Xen) = Inf. First min(Xen)
    # eerst van drie: als Xen[0] == 0 en Xen[len(Xen)-1] != Inf
    if Xen[0] == 0 and Xen[-1] != np.inf:
        if Yen[0] == 0 and Yen[1] != 0:
            # dan loopt er een lijn in de PAV transform tot {inf, -Inf} evenwijdig aan de lijn y=x
            return (np.absolute(np.log10(Xen[1]) - np.log10(Yen[1])))
        else:
            # dan is Yen[0] finite of Yen[1] gelijk 0 en loopt er ergens een horizontale lijn tot Log(Xen[0]) = -Inf. devPAV wordt oneindig
            return np.inf
        # tweede van drie: als Xen[len(Xen)-1] == Inf en Xen[0] != 0
    elif Xen[0] != 0 and Xen[-1] == np.inf:
        if Yen[len(Yen) - 1] == np.inf and Yen[len(Yen) - 2] != np.inf:
            # dan loopt er een lijn in de PAV transform tot {inf, -Inf} evenwijdig aan de lijn y=x
            return (np.absolute(np.log10(Xen[len(Xen) - 2]) - np.log10(Yen[len(Yen) - 2])))
        else:
            # dan is Yen[len(Yen] finite of Yen[len(Yen-2] gelijk inf en loopt er ergens een horizontale lijn tot Log(Xen[len(Xen)]) = Inf. devPAV wordt oneindig
            return np.inf
        # derde van drie: als Xen[0] = 0 en Xen[len(Xen)-1] == Inf
    elif Xen[0] == 0 and Xen[-1] == np.inf:
        if Yen[len(Yen) - 1] == np.inf and Yen[len(Yen) - 2] != np.inf and Yen[0] == 0 and Yen[1] != 0:
            # dan zijn de lijnen aan beide uiteinden evenwijdig met de lijn Y=X. Het berekenen van devPAV is het gemiddelde van twee coordinaten
            devPAV = ((np.absolute(np.log10(Xen[len(Xen) - 2]) - np.log10(Yen[len(Yen) - 2]))) + np.absolute(
                np.log10(Xen[1]) - np.log10(Yen[1]))) / 2
            return (devPAV)
        else:
            # dan loopt er ergens een horizontale lijn met oneindig bereik is is devPAV Inf
            return np.inf

    else:
        # dan is het geen pathological case met rare X-waarden en kan devPAV berekend worden

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
            return (abs(Xen - Yen))
        # than calculate devPAV
        else:
            deltaX = Xen[-1] - Xen[0]
            surface = (0)
            for i in range(1, (len(Xen))):
                surface = surface + calcsurface_f((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
                devPAVs[i - 1] = calcsurface_f((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
            # return(list(surface/a, PAVresult, Xen, Yen, devPAVs))
            return (surface / deltaX)


def devpav(lrs, y):
    """
    calculates PAV transform of LR data under H1 and H2.
    """
    cal = IsotonicCalibrator()
    pavlrs = cal.fit_transform(to_probability(lrs), y)
    return _devpavcalculator(lrs, pavlrs, y)


def calculate_lr_statistics(lr_class0, lr_class1):
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
            avg_llr_class0 = np.mean(np.log2(1 / lr_class0))
            avg_llr_class1 = np.mean(np.log2(lr_class1))
            avg_llr = avg(avg_llr_class0, avg_llr_class1)
        except RuntimeWarning:
            # possibly illegal LRs such as 0 or inf
            avg_llr_class0 = np.nan
            avg_llr_class1 = np.nan
            avg_llr = np.nan

    lrs, y = Xn_to_Xy(lr_class0, lr_class1)
    cllr_class0 = cllr(lrs, y, weights=(1, 0))
    cllr_class1 = cllr(lrs, y, weights=(0, 1))
    cllr_ = .5 * (cllr_class0 + cllr_class1)

    cllrmin_class0 = cllr_min(lrs, y, weights=(1, 0))
    cllrmin_class1 = cllr_min(lrs, y, weights=(0, 1))
    cllrmin = .5 * (cllrmin_class0 + cllrmin_class1)

    return LrStats(avg_llr, avg_llr_class0, avg_llr_class1, avg_p0_class0, avg_p1_class0, avg_p0_class1, avg_p1_class1,
                   cllr_class0, cllr_class1, cllr_, lr_class0, lr_class1, cllrmin, cllr_ - cllrmin)
