import numpy as np

from ..util import to_probability, to_odds, get_classes_from_Xy, Xy_to_Xn


def get_classes_from_scores_Xy(scores, y, classes=None):
    classes = get_classes_from_Xy(scores, y, classes)
    assert scores.shape[1] == classes.size, f'expected: scores has {classes.size} columns; found: {scores.shape[1]}'
    return classes


def scores_Xy_to_Xn(scores, y, classes=[0, 1]):
    classes = get_classes_from_scores_Xy(scores, y, classes)
    return Xy_to_Xn(scores, y, classes)
