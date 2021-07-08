#!/usr/bin/env python3

import argparse
import logging

import lir
import lir.plotting
from lir.data import AlcoholBreathAnalyser


DEFAULT_LOGLEVEL = logging.WARNING

LOG = logging.getLogger(__file__)


class Data:
    def __init__(self, lrs=None, y=None):
        self.lrs = lrs
        self.y = y

    def generate_lrs(self, n):
        gen = lir.generators.NormalGenerator(0., 1., 1., 1.)
        #gen = lir.generators.RandomFlipper(gen, .01)
        self.lrs, self.y = gen.sample_lrs(n//2, n//2)

    def breath_lrs(self):
        self.lrs, self.y = AlcoholBreathAnalyser(ill_calibrated=True).sample_lrs()

    def load_lrs(self, path):
        raise ValueError('not implemented')

    def plot_isotonic(self):
        with lir.plotting.show() as ax:
            ax.pav(self.lrs, self.y)

    def plot_ece(self):
        with lir.plotting.show() as ax:
            ax.ece(self.lrs, self.y)

    def plot_histogram(self):
        with lir.plotting.show() as ax:
            ax.lr_histogram(self.lrs, self.y)

    def plot_nbe(self):
        add_misleading = 1
        print('ELUB', *lir.bayeserror.elub(self.lrs, self.y, add_misleading=add_misleading))
        with lir.plotting.show() as ax:
            ax.nbe(self.lrs, self.y, add_misleading=add_misleading)

    def plot_tippett(self):
        with lir.plotting.show() as ax:
            ax.tippett(self.lrs, self.y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LR operations & plotting')
    plotting = parser.add_argument_group('plotting')
    plotting.add_argument('--plot-isotonic', help='generate an Isotonic Regression plot', action='store_true')
    plotting.add_argument('--plot-ece', help='generate an ECE plot (empirical cross entropy)', action='store_true')
    plotting.add_argument('--plot-histogram', help='generate a histogram', action='store_true')
    plotting.add_argument('--plot-nbe', help='generate an NBE plot (normalized bayes error rate)', action='store_true')
    plotting.add_argument('--plot-tippett', help='generate a Tippett plot', action='store_true')

    etl = parser.add_argument_group('data')
    etl.add_argument('--load-lrs', metavar='FILE', help='read LRs from FILE')
    etl.add_argument('--generate-lrs', metavar='N', type=int, help='draw N LRs from two score distributions')
    etl.add_argument('--breath-lrs', action='store_true', help='use the breath analyser toy data set')

    parser.add_argument('-v', help='increases verbosity', action='count', default=0)
    parser.add_argument('-q', help='decreases verbosity', action='count', default=0)
    args = parser.parse_args()

    data = Data()
    if args.load_lrs:
        data.load_lrs(args.load_lrs)
    if args.generate_lrs:
        data.generate_lrs(args.generate_lrs)
    if args.breath_lrs:
        data.breath_lrs()

    if args.plot_isotonic:
        data.plot_isotonic()
    if args.plot_ece:
        data.plot_ece()
    if args.plot_histogram:
        data.plot_histogram()
    if args.plot_nbe:
        data.plot_nbe()
    if args.plot_tippett:
        data.plot_tippett()
