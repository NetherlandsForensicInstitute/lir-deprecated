"""
Dataset generators.
"""
import numpy as np
from scipy import stats


class NormalGenerator:
    """
    Generate data, can be LRs or scores, for two independent normal distributions: H1 and H2.
    """

    def __init__(self, mu0, sigma0, mu1, sigma1):
        """
        Initializes the generator with two normal distributions.

        :param mu0: mean of class 0 scores (H2)
        :param sigma0: standard deviation of class 0 scores (H2)
        :param mu1: mean of class 1 scores (H1)
        :param sigma1: standard deviation of class 1 scores (H1)
        """
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.mu1 = mu1
        self.sigma1 = sigma1

    def sample_lrs(self, n0, n1):
        """
        Samples LRs in the form P(x|H1)/P(x|H2) for samples from the H2 distribution and the H1 distribution.

        :param n0: number of LRs from class 0 (H2)
        :param n1: number of LRs from class 1 (H1)
        :returns: an array of LRs and an array of labels (value 0 or 1)
        """
        return self._return_probabilities_or_lrs(n0, n1, prob=False)

    def sample_probabilities(self, n0, n1):
        """
        Sample probabilities in the form P(H1) for samples from the H2 distribution and the H1 distribution.

        :param n0: Number of scores from class 0 (H2)
        :param n1: Number of scores from class 1 (H1)
        :return: an array of scores and an array of labels. The scores represent P(H1|x)
        """
        return self._return_probabilities_or_lrs(n0, n1, prob=True)

    def _return_probabilities_or_lrs(self, n0, n1, prob=True):
        """
        Sample probabilities  from both distributions and return either the probabilities  or the o LRs.
        """
        X = np.concatenate([np.random.normal(loc=self.mu0, scale=self.sigma0, size=n0),
                            np.random.normal(loc=self.mu1, scale=self.sigma1, size=n1)])

        p0 = stats.norm.pdf(X, self.mu0, self.sigma0)
        p1 = stats.norm.pdf(X, self.mu1, self.sigma1)

        y = np.concatenate([np.zeros(n0), np.ones(n1)])

        odds = p1 / p0

        return (odds / (1+odds), y) if prob else (odds, y)


class RandomFlipper:
    """
    Random mutilation of a dataset.

    TODO: this class is broken
    """
    def __init__(self, base_generator, p):
        self.gen = base_generator
        self.p = p

    def sample_lrs(self, n0, n1):
        lr, y = self.gen.sample_lrs(n0, n1)
        y[np.random.randint(0, len(y), int(self.p*n0))] = 0
        y[np.random.randint(0, len(y), int(self.p*n1))] = 1

        return lr, y
