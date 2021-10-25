from successor.skaters.scalarskaters.scalarskaterfactory import scaler_skater_factory


def suc_quick_aggressive_ema_ensemble(y,s,k,a=None,t=None,e=None,r=None):
    return scaler_skater_factory(y=y,s=s,k=k,skater_name='quick_aggressive_ema_ensemble',n_input=160)


SCALAR_SIMPLE_SKATERS = [suc_quick_aggressive_ema_ensemble]

