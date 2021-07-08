import collections
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from . import lr, CalibratedScorer, Xy_to_Xn
from .calibration import IsotonicCalibrator
from .metrics import calculate_lr_statistics
from .util import warn_deprecated, Xn_to_Xy


LOG = logging.getLogger(__name__)


def process_vector(preprocessor, *X):
    X = [np.empty(shape=(0, X[0].shape[1])) if x is None else x for x in X]
    X_all = np.concatenate(X)

    X_all = preprocessor.fit_transform(X_all)
    cursor = 0
    X_out = []
    for x in X:
        X_out.append(X_all[cursor:cursor + x.shape[0], :] if x.shape[0] > 0 else None)
        cursor += x.shape[0]

    return X_out


class AbstractCllrEvaluator:
    def __init__(self, name, progress_bar=False):
        self.name = name
        self.progress_bar = progress_bar

    def get_sample(pool, sample_size, default_value):
        if sample_size == -1:
            return pool, None
        elif sample_size is not None and sample_size == pool.shape[0]:
            return pool, None
        elif sample_size is not None:
            sample, newpool = train_test_split(pool, train_size=sample_size)
            assert sample.shape[0] == sample_size
            assert sample.shape[0] + newpool.shape[0] == pool.shape[0]
            return sample, newpool
        else:
            return default_value, pool

    def __call__(self, x,
                 X0=None, X1=None,
                 train_size=None, calibrate_size=None, test_size=None,
                 class0_train_size=None, class0_calibrate_size=None, class0_test_size=None,
                 class1_train_size=None, class1_calibrate_size=None, class1_test_size=None,
                 class0_train=None, class1_train=None, class0_calibrate=None, class1_calibrate=None, class0_test=None,
                 class1_test=None,
                 distribution_mean_delta=None,
                 train_folds=None, train_reuse=False, repeat=1):

        if distribution_mean_delta is not None:
            self._distribution_mean_delta = distribution_mean_delta

        resolve = lambda value: value if value is None or not callable(value) else value(x)

        cllr = []
        for run in tqdm(range(repeat), desc='{} ({})'.format(self.name, x), disable=not self.progress_bar):
            class0_pool = resolve(X0)
            class1_pool = resolve(X1)

            class0_train, class0_pool = AbstractCllrEvaluator.get_sample(class0_pool,
                                                                         class0_train_size if class0_train_size is not None else train_size,
                                                                         resolve(class0_train))
            class0_calibrate, class0_pool = AbstractCllrEvaluator.get_sample(class0_pool,
                                                                             class0_calibrate_size if class0_calibrate_size is not None else calibrate_size,
                                                                             resolve(class0_calibrate))
            class0_test, class0_pool = AbstractCllrEvaluator.get_sample(class0_pool,
                                                                        class0_test_size if class0_test_size is not None else test_size,
                                                                        resolve(class0_test))

            class1_train, class1_pool = AbstractCllrEvaluator.get_sample(class1_pool,
                                                                         class1_train_size if class1_train_size is not None else train_size,
                                                                         resolve(class1_train))
            class1_calibrate, class1_pool = AbstractCllrEvaluator.get_sample(class1_pool,
                                                                             class1_calibrate_size if class1_calibrate_size is not None else calibrate_size,
                                                                             resolve(class1_calibrate))
            class1_test, class1_pool = AbstractCllrEvaluator.get_sample(class1_pool,
                                                                        class1_test_size if class1_test_size is not None else test_size,
                                                                        resolve(class1_test))

            if class0_train is not None and train_folds is not None:
                LOG.debug('evaluate cllr kfold')
                assert class0_train is not None
                assert class1_train is not None
                assert class0_test is not None
                assert class1_test is not None
                cllr.append(self.cllr_kfold(train_folds, class0_train, class1_train, class0_test, class1_test))
            elif class0_calibrate is not None:
                LOG.debug('evaluate cllr')
                cllr.append(
                    self.cllr(class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test))
            elif class0_train is not None and train_reuse:
                LOG.debug('evaluate cllr, reuse training set for calibration')
                cllr.append(self.cllr(class0_train, class1_train, class0_train, class1_train, class0_test, class1_test))

        return cllr


class NormalCllrEvaluator(AbstractCllrEvaluator):
    def __init__(self, name, loc0, scale0, loc1, scale1):
        super().__init__(name)

        self._loc0 = loc0
        self._scale0 = scale0
        self._loc1 = loc1
        self._scale1 = scale1
        self._distribution_mean_delta = None

    def _get_probability(X, mu, sigma):
        return np.exp(-np.power(X - mu, 2) / (2 * sigma * sigma)) / math.sqrt(2 * math.pi * sigma * sigma)

    def _get_lr(self, X):
        # calculate P(E|H0)
        X_p0 = NormalCllrEvaluator._get_probability(X, self._loc0, self._scale0)
        # calculate P(E|H1)
        X_p1 = NormalCllrEvaluator._get_probability(X, self._loc1, self._scale1)
        # calculate LR
        return X_p1 / X_p0

    def cllr_kfold(self, n_splits, X0_train, X1_train, X0_test, X1_test):
        return self.calculate_cllr(X0_test, X1_test)

    def cllr(self, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test):
        return self.calculate_cllr(class0_test, class1_test)

    def calculate_cllr(self, class0_test, class1_test):
        assert class0_test.shape[1] == 1

        # adjust loc1
        if self._distribution_mean_delta is not None:
            self._loc1 = self._loc0 + self._distribution_mean_delta

        # sample from H0
        X0_lr = self._get_lr(class0_test.reshape(-1))
        # sample from H1
        X1_lr = self._get_lr(class1_test.reshape(-1))

        cllr = calculate_lr_statistics(X0_lr, X1_lr)
        return cllr


class ScoreBasedCllrEvaluator(AbstractCllrEvaluator):
    def __init__(self, name, clf, density_function, preprocessors, progress_bar=False):
        super().__init__(name, progress_bar)

        self._clf = clf
        self._pfunc = density_function
        self._preprocessors = preprocessors

    def cllr_kfold(self, n_splits, X0_train, X1_train, X0_test, X1_test):
        for p in self._preprocessors:
            X0_train, X1_train, X0_test, X1_test = process_vector(p, X0_train, X1_train, X0_test, X1_test)
        cllr = lr.scorebased_cllr_kfold(self._clf, self._pfunc, n_splits, X0_train, X1_train, X0_test, X1_test)
        return cllr

    def cllr(self, class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test):
        for p in self._preprocessors:
            class0_train, class1_train, class0_calibrate, class1_calibrate, class0_test, class1_test = process_vector(p,
                                                                                                                      class0_train,
                                                                                                                      class1_train,
                                                                                                                      class0_calibrate,
                                                                                                                      class1_calibrate,
                                                                                                                      class0_test,
                                                                                                                      class1_test)
        cllr = lr.scorebased_cllr(self._clf, self._pfunc, class0_train, class1_train, class0_calibrate,
                                  class1_calibrate, class0_test, class1_test)
        return cllr


class PlotCllrAvg:
    def ylabel():
        return 'C_llr'

    def value(cllr_lst):
        return sum([d.cllr for d in cllr_lst]) / len(cllr_lst)

    def std(cllr_lst):
        return np.std([d.cllr for d in cllr_lst])


class PlotCllrStd:
    def ylabel():
        return 'std(C_llr)'

    def value(cllr_lst):
        return PlotCllrAvg.std(cllr_lst)

    def std(cllr_lst):
        return None


class PlotCllrCal:
    def ylabel():
        return 'C_llr calibration loss'

    def value(cllr_lst):
        return sum([d.cllr_cal for d in cllr_lst]) / len(cllr_lst)

    def std(cllr_lst):
        return None


class PlotLlrAvg:
    def ylabel():
        return 'llr_h0'

    def value(cllr_lst):
        return sum([d.avg_llr_class0 for d in cllr_lst]) / len(cllr_lst)

    def std(cllr_lst):
        return np.std([d.avg_llr_class0 for d in cllr_lst])


class PlotLlrStd:
    def ylabel():
        return 'std(llr_h0)'

    def value(cllr_lst):
        return PlotLlrAvg.std(cllr_lst)

    def std(cllr_lst):
        return None


def makeplot_density(clf, X0_train, X1_train, X0_calibrate, X1_calibrate, calibrators, savefig=None, show=None):
    warn_deprecated()

    line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', ]

    plt.figure(figsize=(20, 20), dpi=100)

    clf.fit(*Xn_to_Xy(X0_train, X1_train))
    points0 = lr.apply_scorer(clf, X0_calibrate)
    points1 = lr.apply_scorer(clf, X1_calibrate)

    for name, f in calibrators:
        f.fit(*Xn_to_Xy(points0, points1))

    x = np.arange(0, 1, .01)

    plt.hist(points0, bins=20, alpha=.25, density=True)
    plt.hist(points1, bins=20, alpha=.25, density=True)

    for i, nf in enumerate(calibrators):
        name, f = nf
        f.transform(x)
        plt.plot(x, f.p0, label=name, c=line_colors[i])
        plt.plot(x, f.p1, label=name, c=line_colors[i])

    plt.legend()

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def makeplot_cllr(xlabel, generators, experiments, savefig=None, show=None,
                  plots=[PlotCllrAvg, PlotCllrStd, PlotCllrCal]):
    warn_deprecated()

    plt.figure(figsize=(20, 20), dpi=100)

    axs = None

    xvalues, _ = zip(*experiments)

    for g in generators:
        LOG.debug('makeplot_cllr: {name}'.format(name=g.name))
        stats = [g(x=x, **genargs) for x, genargs in experiments]

        if axs is None:
            axs = []
            for i, plot in enumerate(plots):
                ax = plt.subplot(len(plots), 1, i + 1)
                plt.ylabel(plot.ylabel())
                axs.append(ax)

            plt.xlabel(xlabel)

        for i in range(len(plots)):
            plot = plots[i]
            ax = axs[i]
            axplot = ax.plot(xvalues, [plot.value(d) for d in stats], 'o--', label=g.name)[0]
            if plot.std(stats[0]) is not None:
                ax.plot(xvalues, [(plot.value(d) - plot.std(d), plot.value(d) + plot.std(d)) for d in stats], '_',
                        color=axplot.get_color())

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, 1), ncol=2)

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def makeplot_accuracy(scorer, density_function, X0_train, X1_train, X0_calibrate, X1_calibrate, title,
                      labels=('class0', 'class1'), savefig=None, show=None):
    warn_deprecated()

    stats = lr.scorebased_cllr(scorer, density_function, X0_train, X1_train, X0_calibrate, X1_calibrate)

    scale = 2

    plt.figure(figsize=(20, 20), dpi=100)

    bins0 = collections.defaultdict(float)
    for v in stats.lr_class0:
        bins0[int(round(math.log(v, scale)))] += (1 if v < 1 else v) / len(stats.lr_class0)

    bins1 = collections.defaultdict(float)
    for v in stats.lr_class1:
        bins1[int(round(math.log(v, scale)))] += (1 if v > 1 else 1 / v) / len(stats.lr_class1)

    bins0_x, bins0_y = zip(*sorted(bins0.items()))
    bins1_x, bins1_y = zip(*sorted(bins1.items()))

    plt.bar(np.array(bins0_x) - .15, bins0_y, label=labels[0], width=.3)
    plt.bar(np.array(bins1_x) + .15, bins1_y, label=labels[1], width=.3)

    plt.title(title)

    plt.legend()

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()


def plot_pav(lrs, y, add_misleading=0, show_scatter=True, savefig=None, show=None, kw_figure={}):
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
        number of misleading evidence points to add on both sides
    show_scatter : boolean
        If True, show individual LRs
    savefig : str
        If not None, write the figure to a file
    show : boolean
        If True, show the plot on screen
    kw_figure : dict
        Keyword arguments that are passed to matplotlib.pyplot.figure()
    ----------
    """
    warn_deprecated()

    pav = IsotonicCalibrator(add_misleading=add_misleading)
    pav_lrs = pav.fit_transform(lrs, y)

    with np.errstate(divide='ignore'):
        llrs = np.log10(lrs)
        pav_llrs = np.log10(pav_lrs)

    fig = plt.figure(**kw_figure)
    xrange = yrange = [llrs[llrs != -np.Inf].min() - .5, llrs[llrs != np.Inf].max() + .5]

    # plot line through origin
    plt.plot(xrange, yrange)

    # line pre pav llrs x and post pav llrs y
    line_x = np.arange(*xrange, .01)
    with np.errstate(divide='ignore'):
        line_y = np.log10(pav.transform(10 ** line_x))

    # filter nan values, happens when values are out of bound (x_values out of training domain for pav)
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
    line_x, line_y = line_x[~np.isnan(line_y)], line_y[~np.isnan(line_y)]

    # some values of line_y go beyond the yrange which is problematic when there are infinite values
    mask_out_of_range = np.logical_and(line_y >= yrange[0], line_y <= yrange[1])
    plt.plot(line_x[mask_out_of_range], line_y[mask_out_of_range])

    # add points for infinite values
    if np.logical_or(np.isinf(pav_llrs), np.isinf(llrs)).any():

        def adjust_ticks_labels_and_range(neg_inf, pos_inf, axis_range):
            ticks = np.linspace(axis_range[0], axis_range[1], 6).tolist()
            tick_labels = [str(round(tick, 1)) for tick in ticks]
            step_size = ticks[2] - ticks[1]

            axis_range = [axis_range[0] - (step_size * neg_inf),axis_range[1] + (step_size * pos_inf)]
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

        plt.yticks(ticks_y, tick_labels_y)
        plt.xticks(ticks_x, tick_labels_x)

        plt.scatter(x_inf,
                    y_inf, facecolors='none', edgecolors='#1f77b4', linestyle=':')

    plt.axis(xrange + yrange)
    # pre-/post-calibrated lr fit

    if show_scatter:
        plt.scatter(llrs, pav_llrs)  # scatter plot of measured lrs

    plt.xlabel("pre-calibrated 10log(lr)")
    plt.ylabel("post-calibrated 10log(lr)")

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()

    plt.close(fig)


def plot_log_lr_distributions_for_model(lr_system: CalibratedScorer, X, y, kind: str = 'histogram', savefig=None,
                                        show=None):
    """
    plots the 10log lrs generated for the two hypotheses by the fitted system when applied to X
    """
    kinds = ['histogram', 'tippett']
    if kind not in kinds:
        raise ValueError(f'kind should be in {kinds}, got {kind}')

    lrs = lr_system.predict_lr(X)
    plot_log_lr_distributions(lrs, y, kind=kind, savefig=savefig, show=show)


def plot_log_lr_distributions(lrs, y, kind: str = 'histogram',
                              savefig=None, show=None, kw_figure={}):
    """
    plots the 10log lrs
    """
    kinds = {
        'histogram': plot_lr_histogram,
        'tippett': plot_tippett,
    }

    if kind in kinds:
        kinds[kind](lrs, y, savefig, show, kw_figure)
    else:
        raise ValueError(f'kind should be in {kinds.keys()}, got {kind}')


def plot_lr_histogram(lrs, y, bins=20, savefig=None, show=None, kw_figure={}):
    """
    plots the 10log lrs
    """
    warn_deprecated()

    plt.figure(**kw_figure)
    log_lrs = np.log10(lrs)

    bins = np.histogram_bin_edges(log_lrs, bins=bins)
    points0, points1 = Xy_to_Xn(log_lrs, y)
    plt.hist(points0, bins=bins, alpha=.25, density=True)
    plt.hist(points1, bins=bins, alpha=.25, density=True)
    plt.xlabel('10log LR')

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()
    plt.close()


def plot_tippett(lrs, y, savefig=None, show=None, kw_figure={}):
    """
    plots the 10log lrs
    """
    warn_deprecated()

    plt.figure(**kw_figure)
    log_lrs = np.log10(lrs)

    xplot = np.linspace(np.min(log_lrs), np.max(log_lrs), 100)
    lr_0, lr_1 = Xy_to_Xn(log_lrs, y)
    perc0 = (sum(i >= xplot for i in lr_0) / len(lr_0)) * 100
    perc1 = (sum(i >= xplot for i in lr_1) / len(lr_1)) * 100

    plt.plot(xplot, perc1, color='b', label='LRs given $\mathregular{H_1}$')
    plt.plot(xplot, perc0, color='r', label='LRs given $\mathregular{H_2}$')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Log likelihood ratio')
    plt.ylabel('Cumulative proportion')
    plt.legend()

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()
    plt.close()


def plot_score_distribution_and_calibrator_fit(calibrator,
                                               scores,
                                               y,
                                               bins=20,
                                               savefig=None,
                                               show=None):
    """
    plots the distributions of scores calculated by the (fitted) lr_system,
    as well as the fitted score distributions/score-to-posterior map
    (Note - for ELUBbounder calibrator is the firststepcalibrator)

    TODO: plot multiple calibrators at once
    """
    warn_deprecated()

    plt.figure(figsize=(10, 10), dpi=100)
    plt.rcParams.update({'font.size': 15})
    bins = np.histogram_bin_edges(scores[np.isfinite(scores)], bins=bins)

    # create weights vector so y-axis is between 0-1
    scores_by_class = [scores[y == cls] for cls in np.unique(y)]
    weights = [np.ones_like(data) / len(data) for data in scores_by_class]

    # adjust weights so largest value is 1
    for i, s in enumerate(scores_by_class):
        hist, _ = np.histogram(s, bins=np.r_[-np.inf, bins, np.inf], weights=weights[i])
        weights[i] = weights[i] * (1 / hist.max())

    x = np.arange(min(bins), max(bins) + 0.01, .01)
    calibrator.transform(x)

    # handle inf values
    if np.isinf(scores).any():
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        x_range = np.linspace(min(bins), max(bins), 6).tolist()
        labels = [str(round(tick, 1)) for tick in x_range]
        step_size = x_range[2] - x_range[1]
        bar_width = step_size / 4
        plot_args_inf = []

        if np.isneginf(scores).any():
            x_range = [x_range[0] - step_size] + x_range
            labels = ['-∞'] + labels
            for i, s in enumerate(scores_by_class):
                if np.isneginf(s).any():
                    plot_args_inf.append(
                        (colors[i], x_range[0] + bar_width if i else x_range[0], np.sum(weights[i][np.isneginf(s)])))

        if np.isposinf(scores).any():
            x_range = x_range + [x_range[-1] + step_size]
            labels.append('∞')
            for i, s in enumerate(scores_by_class):
                if np.isposinf(s).any():
                    plot_args_inf.append(
                        (colors[i], x_range[-1] - bar_width if i else x_range[-1], np.sum(weights[i][np.isposinf(s)])))

        plt.xticks(x_range, labels)

        for color, x_coord, y_coord in plot_args_inf:
            plt.bar(x_coord, y_coord, width=bar_width, color=color, alpha=0.25, hatch='/')

    for cls, weight in zip(np.unique(y), weights):
        plt.hist(scores[y == cls], bins=bins, alpha=.25,
                 label=f'class {cls}', weights=weight)
    plt.plot(x, calibrator.p1, label='fit class 1')
    plt.plot(x, calibrator.p0, label='fit class 0')

    if savefig is not None:
        plt.savefig(savefig)
    if show or savefig is None:
        plt.show()
    plt.close()


class PlottingCalibrator():
    """
    Calibrator wrapper which plots the calibrator fit.

    Usage example:
    ```
    calibrator = lir.plotting.PlottingCalibrator(lir.NormalizedCalibrator(lir.KDECalibrator(bandwidth=.03)), plot_score_distribution_and_calibrator_fit, plot_args={'savefig': 'fig.png'})
    ```
    """

    def __init__(self, calibrator, plot_method, plot_args={}):
        self._calibrator = calibrator
        self._plot_method = plot_method
        self._plot_args = plot_args

    def fit(self, X, y=None, **kwargs):
        self._calibrator.fit(X, y, **kwargs)
        self._plot_method(self._calibrator, X, y, **self._plot_args)

        return self

    def transform(self, X):
        lrs = self._calibrator.transform(X)
        self.p0 = self._calibrator.p0
        self.p1 = self._calibrator.p1
        return lrs
