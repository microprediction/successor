from successor.skaters.scalarskaters.scalarskaterfactory import scaler_skater_factory


def successor_tsa_aggressive_d0_ensemble(y,s,k,a=None,t=None,e=None,r=None):
    return scaler_skater_factory(y=y,s=s,k=k,skater_name='tsa_aggressive_d0_ensemble',n_input=160)


SCALAR_TSA_SKATERS = [successor_tsa_aggressive_d0_ensemble]
