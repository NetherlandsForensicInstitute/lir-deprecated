import os
import unittest


class TestNotebook(unittest.TestCase):
    @unittest.skip("test if the notebook runs without errors")
    def test_execute(self):
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        with open(os.path.join(os.path.dirname(__file__), "..", "practitioners_guide_glass.ipynb")) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {})
