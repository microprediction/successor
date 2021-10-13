# successor ![tests](https://github.com/microprediction/successor/workflows/tests/badge.svg) ![tests-38](https://github.com/microprediction/successor/workflows/tests-38/badge.svg) ![tests-37](https://github.com/microprediction/successor/workflows/tests-37/badge.svg) ![pypi](https://github.com/microprediction/successor/workflows/deploy-pypi/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Uses pre-trained tensorflow models to predict the next k entries in a sequence 

## Install

    pip install successor

You may get better performance by first installing tensorflow following the [instructions](https://www.tensorflow.org/install) and perhaps
reading this [thread](https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags). 

## Use 

See [basic_use](https://github.com/microprediction/successor/tree/main/examples/basic_use.py)

    
    # 1. Import a skater
    from successor.skaters.scalarskaters.scalartsaskaters import successor_tsa_aggressive_d0_ensemble as f

    # 2. Univariate data
    import numpy as np
    y = list(np.cumsum(np.random.randn(1000)))

    # 3. Initialize state to empty dict
    s = {}

    # 4. Give it some data (observations) one at a time, each time passing it back the state s
    for yi in y:
        x, x_std, s = f(y=yi,s=s,k=1)

Skaters follow the convention established by the [timemachines](https://github.com/microprediction/timemachines) library and you are encouraged to read the description of
the "skater" signature if anything is confusing. 

## Benchmarking

See [Elo ratings](https://microprediction.github.io/timeseries-elo-ratings/html_leaderboards/univariate-k_001.html) 

    

