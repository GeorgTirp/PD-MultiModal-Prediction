import numpy as np
from scipy.stats import median_abs_deviation
from numpy import tanh, arctanh
from scipy.special import expit, logit

class RobustTanhScaler:
    def __init__(self, scale_factor=1.4826): #1.4826 for zscore consistency
        self.scale_factor = scale_factor  # for consistency with std if normal
        self.median_ = None
        self.mad_ = None

    def fit(self, y):
        y = np.asarray(y)
        self.median_ = np.median(y)
        self.mad_ = median_abs_deviation(y, scale=self.scale_factor)
        return self

    def transform(self, y):
        y = np.asarray(y)
        if self.median_ is None or self.mad_ is None:
            raise ValueError("Call fit() before transform().")
        z = (y - self.median_) / (self.mad_ + 1e-8)
        return tanh(z)

    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled)
        if self.median_ is None or self.mad_ is None:
            raise ValueError("Call fit() before inverse_transform().")
        z = arctanh(np.clip(y_scaled, -0.999999, 0.999999))  # prevent inf
        return self.median_ + z * self.mad_


class ZScoreScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, y):
        y = np.asarray(y)
        self.mean_ = np.mean(y)
        self.std_ = np.std(y)
        return self

    def transform(self, y):
        y = np.asarray(y)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Call fit() before transform().")
        return (y - self.mean_) / (self.std_ + 1e-8)

    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Call fit() before inverse_transform().")
        return y_scaled * self.std_ + self.mean_


class RobustSigmoidScaler:
    def __init__(self, scale_factor=1.4826):
        self.scale_factor = scale_factor
        self.median_ = None
        self.mad_ = None

    def fit(self, y):
        y = np.asarray(y)
        self.median_ = np.median(y)
        self.mad_ = median_abs_deviation(y, scale=self.scale_factor)
        return self

    def transform(self, y):
        y = np.asarray(y)
        if self.median_ is None or self.mad_ is None:
            raise ValueError("Call fit() before transform().")
        z = (y - self.median_) / (self.mad_ + 1e-8)
        return expit(z)  # sigmoid

    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled)
        if self.median_ is None or self.mad_ is None:
            raise ValueError("Call fit() before inverse_transform().")
        z = logit(np.clip(y_scaled, 1e-6, 1 - 1e-6))
        return self.median_ + z * self.mad_