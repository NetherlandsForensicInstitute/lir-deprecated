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
