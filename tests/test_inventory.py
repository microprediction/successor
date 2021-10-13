import numpy as np


def test_inventory():
    # Ensure we can load and run from JSON
    from successor.skaters.scalarskaters.remote import SKLEARNED_CHAMPIONS, get_remote_compiled_model

    for champ in SKLEARNED_CHAMPIONS:
        model = get_remote_compiled_model(**champ)
        x = np.random.randn(300, 1, champ['n_input'])
        y = model.predict(x)
        assert np.shape(y)==(300,1,1)


if __name__=='__main__':
    test_inventory()

