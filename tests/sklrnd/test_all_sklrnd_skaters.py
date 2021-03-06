from successor.testingutils.skating import run_skater
from pprint import pprint


def test_all_scalar_skaters():
    from successor.skaters.scalarskaters.allscalarskaters import SCALAR_SKATERS
    cpu_metrics = list()
    for k in [8,12]:
        for f in SCALAR_SKATERS:
            x, x_std, s = run_skater(f,k=k)
            if 'cpu' in s:
                cpu_metrics.append( {f.__name__: s['cpu']} )
    pprint(cpu_metrics)


if __name__=='__main__':
    test_all_scalar_skaters()