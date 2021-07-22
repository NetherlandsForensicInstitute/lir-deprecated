#!/usr/bin/env python3

import numpy as np
import unittest

import matplotlib.pyplot as plt

from context import lir
from lir.util import to_probability


class TestPlotting(unittest.TestCase):
    def test_contexts(self):
        lrs = np.array([.5, .5, .5, 1, 1, 2, 2, 2])
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1])

        # inside context
        with lir.plotting.axes() as ax:
            ax.pav(lrs, y)
            ax.title("PAV plot using savefig()")

        # without context
        fig = plt.figure()
        lir.plotting.pav(lrs, y)
        plt.title("simple call with full control")
        plt.close(fig)
        
        # sub plots
        fig, axs = plt.subplots(2)
        lir.plotting.pav(lrs, y, ax=axs[0])
        lir.plotting.ece(lrs, y, ax=axs[1])
        plt.close(fig)

    def test_calls(self):
        lrs = np.array([.5, .5, .5, 1, 1, 2, 2, 2])
        scores = to_probability(lrs)
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1])

        with lir.plotting.axes() as ax:
            ax.pav(lrs, y)

        with lir.plotting.axes() as ax:
            ax.ece(lrs, y)

        with lir.plotting.axes() as ax:
            ax.tippett(lrs, y)

        with lir.plotting.axes() as ax:
            ax.nbe(lrs, y)

        with lir.plotting.axes() as ax:
            ax.lr_histogram(lrs, y)

        with lir.plotting.axes() as ax:
            ax.score_distribution(scores, y)

        cal = lir.LogitCalibrator()
        cal.fit(scores, y)
        with lir.plotting.axes() as ax:
            ax.calibrator_fit(cal)


if __name__ == '__main__':
    unittest.main()
