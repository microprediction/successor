from successor.testingutils.skating import run_skater
from pprint import pprint


def test_all_sklearned_compiled():
    from successor.skaters.scalarskaters.allscalarskaters import SCALAR_SKATERS
    cpu_metrics = list()
    for f in SCALAR_SKATERS:
        x, x_std, s = run_skater(f)
        if 'cpu' in s:
            cpu_metrics.append( {f.__name__: s['cpu']} )
    pprint(cpu_metrics)


if __name__=='__main__':
    test_all_sklearned_compiled()