import numpy as np

# TODO: Move to momentum maybe?


def run_skater(f,k=1):
    s = {}
    y = list(np.random.randn(500))
    for yi in y:
        x, x_std, s = f(y=yi,s=s,k=k)
    return x, x_std, s
