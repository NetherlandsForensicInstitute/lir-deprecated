import numpy as np
import matplotlib.pyplot as plt
from lir import IsotonicCalibrator

# make matplotlib.pyplot behave more like axes objects
plt.set_xlabel = plt.xlabel
plt.set_ylabel = plt.ylabel
plt.set_xlim = plt.xlim
plt.set_ylim = plt.ylim


def pav_zoom(lrs, y, add_misleading=0, show_scatter=True, ax=plt):
    """
    Generates a plot of pre- versus post-calibrated LRs using Pool Adjacent
    Violators (PAV).

    Parameters
    ----------
    lrs : numpy array of floats
        Likelihood ratios before PAV transform
    y : numpy array
        Labels corresponding to lrs (0 for Hd and 1 for Hp)
    add_misleading : int
        number of misleading evidence points to add on both sides (default: `0`)
    show_scatter : boolean
        If True, show individual LRs (default: `True`)
    ax : pyplot axes object
        defaults to `matplotlib.pyplot`
    ----------
    """
    pav = IsotonicCalibrator(add_misleading=add_misleading)
    pav_lrs = pav.fit_transform(lrs, y)

    with np.errstate(divide='ignore'):
        llrs = np.log10(lrs)
        pav_llrs = np.log10(pav_lrs)

    xrange = yrange = [min(min(pav_llrs[pav_llrs != -np.Inf]), min(llrs[pav_llrs != -np.Inf])) - 0.5,
                       max(max(pav_llrs[pav_llrs != np.Inf]), max(llrs[pav_llrs != np.Inf])) + 0.5]

    # plot line through origin
    ax.plot(xrange, yrange)

    # line pre pav llrs x and post pav llrs y
    line_x = np.arange(*xrange, .01)
    with np.errstate(divide='ignore'):
        line_y = np.log10(pav.transform(10 ** line_x))

    # filter nan values, happens when values are out of bound (x_values out of training domain for pav)
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
    line_x, line_y = line_x[~np.isnan(line_y)], line_y[~np.isnan(line_y)]

    # some values of line_y go beyond the yrange which is problematic when there are infinite values
    mask_out_of_range = np.logical_and(line_y >= yrange[0], line_y <= yrange[1])
    ax.plot(line_x[mask_out_of_range], line_y[mask_out_of_range])

    # add points for infinite values
    if np.logical_or(np.isinf(pav_llrs), np.isinf(llrs)).any():
        def adjust_ticks_labels_and_range(neg_inf, pos_inf, axis_range):
            ticks = np.linspace(axis_range[0], axis_range[1], 6).tolist()
            tick_labels = [str(round(tick, 1)) for tick in ticks]
            step_size = ticks[2] - ticks[1]

            axis_range = [axis_range[0] - (step_size * neg_inf), axis_range[1] + (step_size * pos_inf)]
            ticks = [axis_range[0]] * neg_inf + ticks + [axis_range[1]] * pos_inf
            tick_labels = ['-∞'] * neg_inf + tick_labels + ['+∞'] * pos_inf

            return axis_range, ticks, tick_labels

        def replace_values_out_of_range(values, min_range, max_range):
            # create margin for point so no overlap with axis line
            margin = (max_range - min_range) / 60
            return np.concatenate([np.where(np.isneginf(values), min_range + margin, values),
                                   np.where(np.isposinf(values), max_range - margin, values)])

        yrange, ticks_y, tick_labels_y = adjust_ticks_labels_and_range(np.isneginf(pav_llrs).any(),
                                                                       np.isposinf(pav_llrs).any(),
                                                                       yrange)
        xrange, ticks_x, tick_labels_x = adjust_ticks_labels_and_range(np.isneginf(llrs).any(),
                                                                       np.isposinf(llrs).any(),
                                                                       xrange)

        mask_not_inf = np.logical_or(np.isinf(llrs), np.isinf(pav_llrs))
        x_inf = replace_values_out_of_range(llrs[mask_not_inf], xrange[0], xrange[1])
        y_inf = replace_values_out_of_range(pav_llrs[mask_not_inf], yrange[0], yrange[1])

        ax.yticks(ticks_y, tick_labels_y)
        ax.xticks(ticks_x, tick_labels_x)

        ax.scatter(x_inf,
                   y_inf, facecolors='none', edgecolors='#1f77b4', linestyle=':')

    ax.axis(xrange + yrange)
    # pre-/post-calibrated lr fit

    if show_scatter:
        ax.scatter(llrs, pav_llrs)  # scatter plot of measured lrs

    ax.set_xlabel("pre-calibrated log$_{10}$(LR)")
    ax.set_ylabel("post-calibrated log$_{10}$(LR)")

"""
Empirical Cross Entrpy (ECE)

The discrimination and calibration of the LRs reported by some systems can also
be measured separately. The empirical cross entropy (ECE) plot is a graphical
way of doing this.

The ECE is the average of -P(Hp) * log2(P(Hp|LRi)) for all LRi when Hp is true,
and -P(Hd) * log2(P(Hd|LRi)) for all LRi when Hd is true.

See:
[-] D. Ramos, Forensic evidence evaluation using automatic speaker recognition
    systems. Ph.D. Thesis. Universidad Autonoma de Madrid.
[-] Bernard Robertson, G.A. Vignaux and Charles Berger, Interpreting Evidence:
    Evaluating Forensic Science in the Courtroom, 2nd edition, 2016, pp. 96-97.
"""
# import matplotlib.pyplot as plt
# import numpy as np

from lir import calibration, util
from lir.util import warn_deprecated


def plot(lrs, y, log_prior_odds_range=None, on_screen=False, path=None, kw_figure={}):
    warn_deprecated()

    fig = plt.figure(**kw_figure)
    plot_ece_zoom(lrs, y, log_prior_odds_range)

    if on_screen:
        plt.show()
    if path is not None:
        plt.savefig(path)

    plt.close(fig)


def plot_ece_zoom(lrs, y, log_prior_odds_range=None, ax=plt):
    """
    Generates an ECE plot for a set of LRs and corresponding ground-truth
    labels.

    The x-axis indicates the log prior odds of a sample being drawn from class
    1; the y-axis shows the entropy for (1) a non-informative system (dotted
    line), (2) the set of LR values (line), and (3) the set of LR values after
    PAV-transformation (Pool Adjacent Violators, dashed line).

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param log_prior_odds_range: the range of prior odds (tuple of two values,
        indicating both ends of the range on the x-axis)
    """
    if log_prior_odds_range is None:
        log_prior_odds_range = (-3, 3)

    log_prior_odds = np.arange(*log_prior_odds_range, .01)
    prior_odds = np.power(10, log_prior_odds)

    # plot reference
    ax.plot(log_prior_odds, calculate_ece(np.ones(len(lrs)), y, util.to_probability(prior_odds)), linestyle=':', label='reference')

    LRs_ece = calculate_ece(lrs, y, util.to_probability(prior_odds))
    ylim = max(LRs_ece)*1.5

    # plot LRs
    ax.plot(log_prior_odds, LRs_ece, linestyle='-', label='LRs')

    # plot PAV LRs
    pav_lrs = calibration.IsotonicCalibrator().fit_transform(util.to_probability(lrs), y)
    ax.plot(log_prior_odds, calculate_ece(pav_lrs, y, util.to_probability(prior_odds)), linestyle='--', label='PAV LRs')

    ax.set_xlabel("prior log$_{10}$(odds)")
    ax.set_ylabel("empirical cross-entropy")
    ax.set_ylim((0, ylim))
    ax.set_xlim(log_prior_odds_range)
    ax.legend()
    ax.grid(True, linestyle=':')


def calculate_ece(lrs, y, priors):
    """
    Calculates the empirical cross-entropy (ECE) of a set of LRs and
    corresponding ground-truth labels.

    An entropy is calculated for each element of `priors`.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels of the LRs (values 0 for Hd or 1
        for Hp); must be of the same length as `lrs`.
    :param priors: an array of prior probabilities of the samples being drawn
        from class 1 (values in range [0..1])
    :returns: an array of entropy values of the same length as `priors`
    """
    assert np.all(lrs >= 0), "invalid input for LR values"
    assert np.all(np.unique(y) == np.array([0, 1])), "label set must be [0, 1]"

    prior_odds = np.repeat(util.to_odds(priors), len(lrs)).reshape((len(priors), len(lrs)))
    posterior_odds = prior_odds * lrs
    posterior_p = util.to_probability(posterior_odds)

    with np.errstate(divide='ignore'):
        ece0 = - (1 - priors.reshape((len(priors),1))) * np.log2(1 - posterior_p[:,y == 0])
        ece1 = -      priors.reshape((len(priors),1))  * np.log2(    posterior_p[:,y == 1])

    ece0[np.isnan(ece0)] = np.inf
    ece1[np.isnan(ece1)] = np.inf

    avg0 = np.average(ece0, axis=1)
    avg1 = np.average(ece1, axis=1)

    return avg0 + avg1
