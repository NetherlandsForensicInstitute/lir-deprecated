import os
import numpy as np


class AlcoholBreathAnalyser:
    """
    Example from paper:
        Peter Vergeer, Andrew van Es, Arent de Jongh, Ivo Alberink and Reinoud
        Stoel, Numerical likelihood ratios outputted by LR systems are often
        based on extrapolation: When to stop extrapolating? In: Science and
        Justice 56 (2016) 482–491.
    """
    def __init__(self, ill_calibrated=False):
        self.ill_calibrated = ill_calibrated

    def sample_lrs(self):
        positive_lr = 1000 if self.ill_calibrated else 90
        lrs = np.concatenate([np.ones(990) * 0.101, np.ones(10) * positive_lr, np.ones(90) * positive_lr, np.ones(10) * .101])
        y = np.concatenate([np.zeros(1000), np.ones(100)])
        return lrs, y

class UnboundedLRs:
    """"
    Examples from paper:
        A transparent method to determine limit values for Likelihood Ratio systems, by
        Ivo Alberink, Jeannette Leegwater, Jonas Malmborg, Anders Nordgaard, Marjan Sjerps, Leen van der Ham
        In: Submitted for publication in 2025.
    """
    def __init__(self, example=4):
        self.example = example

    def sample_lrs(self):
        if self.example == 4:
            np.random.seed(42)
            lrs_h1 = np.exp(np.random.normal(1, 1, 100))
            lrs_h2 = np.exp(np.random.normal(0, 1, 1000))
        elif self.example == 5:
            dirname = os.path.dirname(__file__)
            input_path = os.path.join(dirname, 'resources/lr_bounding')
            llrs_h1 = np.loadtxt(os.path.join(input_path, 'LLR_KM.csv'))
            llrs_h2 = np.loadtxt(os.path.join(input_path, 'LLR_KNM.csv'))
            lrs_h1 = np.power(10, llrs_h1)
            lrs_h2 = np.power(10, llrs_h2)
        else:
            raise ValueError('Only examples 4 and 5 are supported.')
        lrs = np.append(lrs_h1, lrs_h2)
        y = np.append(np.ones((len(lrs_h1), 1)), np.zeros((len(lrs_h2), 1)))
        return lrs, y
