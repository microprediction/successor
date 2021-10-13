
import numpy as np


def nan_helper(y):
    # Lazy C&P https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    return np.isnan(y), lambda z: z.nonzero()[0]


def linear_interpolator(y):
    if len(y)==1:
        return y
    else:
        nans, x = nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
        return y