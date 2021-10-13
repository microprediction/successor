
import numpy as np


def nan_helper(y):
    # Lazy C&P https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    return np.isnan(y), lambda z: z.nonzero()[0]


def linear_interpolator(y):
    if len(y)==1:
        return y
    else:
        return interp_nans(y)


def interp_nans(x:[float],left=None, right=None, period=None)->[float]:
    """ [1 1 1 nan nan 2 2 nan 0] -> [1 1 1 1.3 1.6 2 2  1  0]
        Same conventions as https://numpy.org/doc/stable/reference/generated/numpy.interp.html
    """
    xp = [i for i, yi in enumerate(x) if np.isfinite(yi)]
    fp = [yi for i, yi in enumerate(x) if np.isfinite(yi)]
    return list(np.interp(x=list(range(len(x))), xp=xp, fp=fp,left=left,right=right,period=period))