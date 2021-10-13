

def reflect(y:[float], n_input:int):
    """ Prepend to start of vector """
    if len(y)>=n_input:
        return y
    else:
        assert len(y),'Cannot extend zero length vector'
        y_extended = [c for c in y]
        while len(y_extended)<n_input:
            y_extended = list(reversed(y_extended)) + y_extended
        return y_extended[-n_input:]




