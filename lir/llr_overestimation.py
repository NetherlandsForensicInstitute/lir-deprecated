import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from KDEpy import NaiveKDE

def plot_llr_overestimation(
        lrs: np.array, y: np.array, num_fids: int = 1000, ax: plt.Axes = plt, **kwargs):
    """
    Plot the LLR-overestimation as function of the system LLR.

     The LLR-overestimation is defined as the log-10 of the ratio between
        (1) the system LRs; the outputs of the LR-system, and
        (2) the empirical LRs; the ratio's between the relative frequencies of the H1-LLRs and H2-LLRs.

     - See documentation on :func:`calc_llr_overestimation` for more details on the LLR-overestimation.
     - An interval around the LLR-overestimation can be calculated using fiducial distributions.
     - The average absolute LLR-overestimation can be used as single metric.

    :param lrs: the likelihood ratios (LRs), as calculated by the LR-system
    :param y: the corresponding labels (0 for H2 or Hd, 1 for H1 or Hp)
    :param num_fids: number of fiducial distributions to base the interval on; use 0 for no interval
    :param ax: matplotlib axes to plot into
    :param kwargs: additional arguments to pass to :func:`calc_llr_overestimation` and/or :func:`calc_fiducial_density_functions`
    """

    # Calculate all required numbers
    llr_grid, llr_overestimation, llr_overestimation_interval = calc_llr_overestimation(lrs, y, num_fids, **kwargs)
    llr_misestimation = np.mean(np.abs(llr_overestimation))

    # Make the LLR-overestimation plot
    label_text = 'mean(abs) = ' + str(np.round(llr_misestimation, 2))
    line = ax.plot(llr_grid, llr_overestimation, label=label_text)
    if num_fids > 0:
        ax.plot(llr_grid, llr_overestimation_interval[:,(0,2)], linestyle=':', color=line[0].get_color())
    ax.axhline(color='k', linestyle='dotted')
    ax.set_xlabel('log$_{10}$(system LR)')
    ax.set_ylabel('LLR-overestimation')
    ax.legend(loc="upper center")

def calc_llr_overestimation(
        lrs: np.array, y: np.array, num_fids: int = 1000, bw: tuple[str|float,str|float] = ('silverman','silverman'),
         num_grid_points: int = 100, alpha: float = 0.05, **kwargs) -> tuple[np.array, np.array, np.array]:
    """
    Calculate the LLR-overestimation as function of the system LLR.

     The LLR-overestimation is defined as the log-10 of the ratio between
        (1) the system LRs; the outputs of the LR-system, and
        (2) the empirical LRs; the ratio's between the relative frequencies of the H1-LLRs and H2-LLRs.

     - It quantifies the deviation from the requirement that 'the LR of the LR is the LR': the 'LR-consistency'.
     - For a perfect LR-system, the LLR-overestimation is 0: the system and empirical LRs are the same.
     - A positive LLR-overestimation indicates that the system LRs are too high, compared to the empirical LRs.
     - An LLR-overestimation of +1 indicates that the system LRs are too high by a factor of 10.
     - An LLR-overestimation of -1 indicates that the system LRs are too low by a factor of 10.
     - The relative frequencies are estimated with KDE using Silverman's rule-of-thumb for the bandwidths.
     - An interval around the LLR-overestimation can be calculated using fiducial distributions.

    :param lrs: the likelihood ratios (LRs), as calculated by the LR-system
    :param y: the corresponding labels (0 for H2 or Hd, 1 for H1 or Hp)
    :param num_grid_points: number of points used in the grid to calculate the LLR-overestimation on
    :param bw: two bandwidths for the KDEs of H1 & H2; for each specify a method (string) or a value (float)
    :param num_fids: number of fiducial distributions to base the interval on; use 0 for no interval
    :param alpha: level of confidence to use for the interval
    :param kwargs: additional arguments to pass to :func:`calc_fiducial_density_functions`
    """

    # Convert the LRs to log10 values (LLRs)
    llr_h1 = np.log10(lrs[y==1])
    llr_h2 = np.log10(lrs[y==0])
    # Determine x-limits to be used in the plot: based on lowest H1-LLRs and highest H2-LLRs
    # In case of very many LLRs do not take the very extreme values, due to instabilities in areas with little data
    llr_min = np.max((np.quantile(llr_h1, 0.001, method='hazen'), np.quantile(llr_h2, 0.001, method='hazen')))
    llr_max = np.min((np.quantile(llr_h1, 0.999, method='hazen'), np.quantile(llr_h2, 0.999, method='hazen')))
    if llr_min >= llr_max:
        print('There is no overlap between the H1- and H2-distributions: LLR-overestimation cannot be determined.')
        return np.nan, np.nan, np.nan
    llr_grid = np.linspace(llr_min, llr_max, num_grid_points)
    # Get the pdf's of the empirical distributions for the two LLR-sets
    # If a method is specified to determine the bandwidth, then do this based on the data within the relevant range only
    if isinstance(bw[0], str):
        llr_inrange = (llr_h1 >= llr_min) & (llr_h1 <= llr_max)
        kde_h = NaiveKDE(bw=bw[0]).fit(llr_h1[llr_inrange])
        bw_h1 = kde_h.bw
    else:
        bw_h1 = bw[0]
    if isinstance(bw[1], str):
        llr_inrange = (llr_h2 >= llr_min) & (llr_h2 <= llr_max)
        kde_h = NaiveKDE(bw=bw[1]).fit(llr_h2[llr_inrange])
        bw_h2 = kde_h.bw
    else:
        bw_h2 = bw[1]
    # Apply the bandwidth to the full dataset when calculating the probability densities.
    pdf_empirical_h1 = NaiveKDE(bw=bw_h1).fit(llr_h1).evaluate(llr_grid)
    pdf_empirical_h2 = NaiveKDE(bw=bw_h2).fit(llr_h2).evaluate(llr_grid)
    # Calculate the empirical LRs
    lrs_empirical = pdf_empirical_h1 / pdf_empirical_h2
    # Use them to calculate the LLR-overestimation
    lrs_system = 10**llr_grid
    llr_overestimation = np.log10(lrs_system / lrs_empirical)
    # Also calculate an interval around the LLR-overestimation
    if num_fids > 0:
        # Calculate pdf's of fiducial distributions of the two LLR-sets, and calculate distributions of observed LRs
        pdfs_empirical_fid_h1 = calc_fiducial_density_functions(llr_h1, llr_grid, 'pdf', num_fids, **kwargs)
        pdfs_empirical_fid_h2 = calc_fiducial_density_functions(llr_h2, llr_grid, 'pdf', num_fids, **kwargs)
        pdfs_empirical_fid_h1[pdfs_empirical_fid_h1 == 0] = np.nan
        pdfs_empirical_fid_h2[pdfs_empirical_fid_h2 == 0] = np.nan
        lrs_empirical_fid = pdfs_empirical_fid_h1 / pdfs_empirical_fid_h2
        # Calculate percentiles of these distributions; allowing some nans, but not too many
        percentages = (100 * alpha/2, 50, 100 *  (1 - alpha/2))
        lrs_empirical_interval = np.nanpercentile(lrs_empirical_fid, percentages, axis=1, method='hazen').transpose()
        too_many_nans = np.sum(np.isnan(lrs_empirical_fid), axis=1) > (0.05 * num_fids)
        lrs_empirical_interval[too_many_nans,:] = np.nan
        # Do the conversion to the LLR-overestimation
        lrs_system_2d = np.tile(np.expand_dims(lrs_system, 1), (1, len(percentages)))
        llr_overestimation_interval = np.log10(lrs_system_2d / lrs_empirical_interval)
    else:
        llr_overestimation_interval = np.nan
    return llr_grid, llr_overestimation, llr_overestimation_interval

def calc_fiducial_density_functions(data: np.array, grid: np.array, df_type: str ='pdf', num_fids: int = 1000,
                                    smoothing_grid_fraction: float = 1/10, smoothing_sample_size_correction: float = 1,
                                    seed: [None|int] = None) -> np.array:
    """
    Calculate (smoothed) density functions of fiducial distributions of a dataset.

    :param data: 1-dimensional array of data points
    :param grid: 1-dimensional array of equally spaced grid points, at which to calculate the density functions
    :param df_type: type of density function (df) to generate: either probability ('pdf') or cumulative ('cdf')
    :param num_fids: number of fiducial distributions to generate
    :param smoothing_grid_fraction: fraction of grid points to use as half window during smoothing
    :param smoothing_sample_size_correction: value to use for sample size correction of smoothing window; 0 is no correction
    :param seed: seed for random number generator used draw samples from a uniform distribution
    """

    # Generate cdfs of the fiducial distributions: sorted random draws from a uniform distribution
    rng = np.random.default_rng(seed)
    cdfs = np.sort(rng.uniform(size=(len(data), num_fids)), axis=0)
    # Savitzky-Golay filtering will be used later on during this calculation. To try and limit edge-effects near the
    # grid edges, an extended grid will be used. Half of the filter window will be added at each side of the grid.
    # The size (number of grid points) of the filter window depends on the grid size and on the sample size. The
    # fraction of the grid to use as half window is specified by the 'smoothing_grid_fraction' parameter; the sample
    # size correction (if any) by the 'smoothing_sample_size_correction' parameter. For a sample size around 100, it is
    # optimal to use about 10% of the number of grid points as half window; this is used as basis for the correction.
    # For a sample size of 10 the percentage increases to 20%, and for a sample size of 1000 it decreases to about 5%.
    samples_in_grid = np.sum((data >= np.min(grid)) & (data <= np.max(grid)))
    sample_size_correction = 2**(np.log10(100 / samples_in_grid) * smoothing_sample_size_correction)
    half_window = int(len(grid) * smoothing_grid_fraction * sample_size_correction)
    grid_diff = float(np.diff(grid[:2]))
    extended_grid = np.concatenate((np.linspace(grid[0] - half_window * grid_diff, grid[0] - grid_diff, half_window),
                              grid, np.linspace(grid[-1] + grid_diff, grid[-1] + half_window * grid_diff, half_window)))
    # Do interpolation on this grid; make sure extrapolation is allowed (typically near probabilities of 0 and 1)
    f_interp = interp1d(np.sort(data), cdfs, axis=0, kind='linear', fill_value='extrapolate')
    cdfs_extended_grid = f_interp(extended_grid)
    # Ensure no probabilities below 0 or above 1 in the cdfs
    cdfs_extended_grid[cdfs_extended_grid < 0] = 0
    cdfs_extended_grid[cdfs_extended_grid > 1] = 1
    # Calculate pdfs (1st derivative) or cdfs (no derivative) , while also performing smoothing/noise-reduction;
    # This uses a low-order Savitzky-Golay filter, which requires an equally spaced grid.
    window_length = 2 * half_window + 1
    if df_type == 'pdf':
        derivative_order = 1
    elif df_type == 'cdf':
        derivative_order = 0
    else:
        raise ValueError('Unsupported type of density function specified: only cdf and pdf are supported.')
    dfs_extended_grid = savgol_filter(cdfs_extended_grid, window_length, polyorder=2, axis=0,
                                       deriv=derivative_order, delta=grid_diff, mode='nearest')
    # Ensure no probabilities below 0 are present
    dfs_extended_grid[dfs_extended_grid < 0] = 0
    dfs_grid = dfs_extended_grid[half_window:-half_window]
    return dfs_grid
