LIR Python Likelihood Ratio Library
===================================

This library provides a collection of scripts to aid calibration, and
calculation and evaluation of Likelihood Ratios.

## A simple score-based LR system

A score-based LR system needs a scorer and a calibrator. The most basic setup
uses a training set and a test set. Both the scorer and the calibrator are
fitted on the training set.

```python
import lir
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# generate some data randomly from a normal distribution
X = np.concatenate([np.random.normal(loc=0, size=(100, 1)),
              np.random.normal(loc=1, size=(100, 1))])
y = np.concatenate([np.zeros(100), np.ones(100)])

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# initialize a scorer and a calibrator
scorer = LogisticRegression(solver='lbfgs')  # choose any sklearn style classifier
calibrator = lir.KDECalibrator()  # use plain KDE for calibration
calibrated_scorer = lir.CalibratedScorer(scorer, calibrator)

# fit and predict
calibrated_scorer.fit(X_train, y_train)
lrs_test = calibrated_scorer.predict_lr(X_test)

# print the quality of the system as log likelihood ratio cost (lower is better)
print('The log likelihood ratio cost is', lir.metrics.cllr(lrs_test, y_test), '(lower is better)')
print('The discriminative power is', lir.metrics.cllr_min(lrs_test, y_test), '(lower is better)')

# plot calibration
import lir.plotting
with lir.plotting.show() as ax:
    ax.pav(lrs_test, y_test)
```

The log likelihood ratio cost (CLLR) may be used as a metric of performance.
In this case it should yield a value of around .8, but highly variable due to
the small number of samples. Increase the sample size to get more stable
results.
