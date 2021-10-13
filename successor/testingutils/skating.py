import numpy as np


def run_skater(f):
    s = {}
    y = list(np.random.randn(500))
    for yi in y:
        x, x_std, s = f(y=yi,s=s,k=1)
    return x, x_std, s
