from pprint import pprint

# Illustrates how to use "skater" forecasting functions provided in this package

if __name__=='__main__':
    import numpy as np

    # 1. Import a skater
    from successor.skaters.scalarskaters.scalartsaskaters import suc_tsa_p2_d0_q1 as f

    # 2. Univariate data
    y = list(np.cumsum(np.random.randn(1000)))

    # 3. Initialize state to empty dict
    s = {}

    # 4. Give it some data (observations) one at a time, each time passing it back the state s
    for yi in y:
        x, x_std, s = f(y=yi,s=s,k=1)

    # Print performance statistics
    pprint(s.get('cpu'))


