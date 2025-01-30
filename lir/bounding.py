"""
Extrapolation bounds on LRs using the method by Alberink et al. (2025)

See:
[-] Ivo Alberink, Jeannette Leegwater, Jonas Malmborg, Anders Nordgaard, Marjan Sjerps, Leen van der Ham
    A transparent method to determine limit values for Likelihood Ratio systems
    In: to be submitted for publication.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_delta_functions(lrs: np.ndarray, y: np.ndarray, llr_threshold_range: tuple[float, float] = None,
                         step_size: float = .001, ax: plt.Axes = plt) -> None:
    """
    Returns a figure of the delta functions along with the upper and lower bounds of the LRs.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param llr_threshold_range: lower limit and upper limit for the LLRs to include in the figure
    :param step_size: required accuracy on a base-10 logarithmic scale
    :param ax: matplotlib axes
    """

    if llr_threshold_range is None:
        llrs = np.log10(lrs)
        llr_threshold_range = (np.min(llrs) - .5, np.max(llrs) + .5)

    llr_threshold = np.arange(*llr_threshold_range, step_size)

    lower_bound, upper_bound, delta_low, delta_high = calculate_bounds(lrs, y, llr_threshold=llr_threshold)

    # plot the delta-functions and the 0-line
    lower_llr = np.round(np.log10(lower_bound),2)
    upper_llr = np.round(np.log10(upper_bound),2)
    ax.plot(llr_threshold, delta_low, '--', label=r"$\Delta_{lower}$ is 0 at " + str(lower_llr))
    ax.plot(llr_threshold, delta_high, '-', label=r"$\Delta_{upper}$ is 0 at " + str(upper_llr))
    ax.axhline(color='k', linestyle='dotted')
    # Some more formatting
    ax.legend(loc="upper left")
    ax.xlabel("log10(LR)")
    ax.ylabel("$\Delta$-value")

def calculate_bounds(lrs: np.ndarray, y: np.ndarray, llr_threshold: np.ndarray = None, step_size: float = .001,
                     substitute_extremes: tuple[float, float] = (np.exp(-20), np.exp(20))) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns the upper and lower bounds of the LRs.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param llr_threshold: predefined values of LLRs as possible bounds
    :param step_size: required accuracy on a base-10 logarithmic scale
    :param substitute_extremes: (tuple of scalars) substitute for extreme LRs, i.e.
        LRs of 0 and inf are substituted by these values
    """

    # remove LRs of 0 and infinity
    sanitized_lrs = lrs
    sanitized_lrs[sanitized_lrs < substitute_extremes[0]] = substitute_extremes[0]
    sanitized_lrs[sanitized_lrs > substitute_extremes[1]] = substitute_extremes[1]

    # determine the range of LRs to be considered
    if llr_threshold is None:
        llrs = np.log10(sanitized_lrs)
        llr_threshold_range = (min(0, np.min(llrs)), max(0, np.max(llrs))+step_size)
        llr_threshold = np.arange(*llr_threshold_range, step_size)

    # calculate the two delta functions
    delta_low, delta_high = calculate_delta_functions(lrs, y, llr_threshold)

    # find the LLRs closest to LLR=0 where the functions become negative & convert them to LRs
    # if no negatives are found, use the maximum H1-LR in case of upper bound & minimum H2-LR in case of lower bound
    delta_high_negative = np.where(delta_high < 0)[0]
    if not any(delta_high_negative):
        upper_bound = np.max(lrs[y==1])
    else:
        pst_upper_bound = delta_high_negative[0] - 1
        upper_bound = 10 ** llr_threshold[pst_upper_bound]
    delta_low_negative = np.where(delta_low < 0)[0]
    if not any(delta_low_negative):
        lower_bound = np.min(lrs[y==0])
    else:
        pst_lower_bound = delta_low_negative[-1] + 1
        lower_bound = 10 ** llr_threshold[pst_lower_bound]

    # Check for bounds on the wrong side of 1. This may occur for badly
    # performing LR systems, e.g. if the delta function is always below zero.
    lower_bound = min(lower_bound, 1)
    upper_bound = max(upper_bound, 1)

    return lower_bound, upper_bound, delta_low, delta_high

def calculate_delta_functions(lrs: np.ndarray, y: np.ndarray, llr_threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the delta functions for a set of LRs at given threshold values.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param llr_threshold: an array of threshold LLRs
    :returns: two arrays of delta-values, at all threshold LR values
    """

    # fix the value used for the beta distributions at 1/2 (Jeffreys prior)
    beta_parameter = 1 / 2

    # for all possible llr_threshold values, count how many of the lrs are larger or equal to them for both h1 and h2
    lrs_h1 = lrs[y==1]
    lrs_h2 = lrs[y==0]
    llr_h1_2d = np.tile(np.expand_dims(np.log10(lrs_h1), 1), (1, llr_threshold.shape[0]))
    llr_h2_2d = np.tile(np.expand_dims(np.log10(lrs_h2), 1), (1, llr_threshold.shape[0]))
    success_h1 = np.sum(llr_h1_2d >= llr_threshold, axis=0)
    success_h2 = np.sum(llr_h2_2d >= llr_threshold, axis=0)

    # use the as inputs for calculations of the probabilities
    prob_h1_above_grid = (success_h1 + beta_parameter) / (len(lrs_h1) + 2*beta_parameter)
    prob_h2_above_grid = (success_h2 + beta_parameter) / (len(lrs_h2) + 2*beta_parameter)
    prob_h1_below_grid = 1 - prob_h1_above_grid
    prob_h2_below_grid = 1 - prob_h2_above_grid

    # calculate the delta-functions for all the llr_threshold values
    delta_high = np.log10(prob_h1_above_grid) - np.log10(prob_h2_above_grid) - llr_threshold
    delta_low = llr_threshold - np.log10(prob_h1_below_grid) + np.log10(prob_h2_below_grid)

    return delta_low, delta_high
