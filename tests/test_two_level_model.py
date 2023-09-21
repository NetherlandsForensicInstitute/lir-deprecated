import unittest

import numpy as np

from lir.classifiers.two_level_model import TwoLevelModel


class TestTwoLevelModel(unittest.TestCase):

    two_level_model = TwoLevelModel()
    # TODO read necessary files for tests, implement tests

    def test_dummy_fit(self):
        """
        TODO dummy test, remove when we have an actual test
        """
        self.two_level_model.fit(X=None, y=None)
        prediction = self.two_level_model.predict(np.array([5]))
        np.testing.assert_equal(prediction, 0.5)


if __name__ == '__main__':
    unittest.main()
