from successor.skaters.scalarskaters.scalarskaterfactory import scaler_skater_factory


def suc_tsa_p2_d0_q1(y,s,k,a=None,t=None,e=None,r=None):
    return scaler_skater_factory(y=y,s=s,k=k,skater_name='tsa_p2_d0_q1',n_input=160)


SCALAR_TSA_SKATERS = [suc_tsa_p2_d0_q1]

