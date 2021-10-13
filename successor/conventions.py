# Borrowed from timemachines

def nonecast(x,fill_value=0.0):
    if x is not None:
        return [xj if xj is not None else fill_value for xj in x]

def wrap(x):
    """ Ensure x is a list of float """
    if x is None:
        return None
    elif isinstance(x,(float,int)):
        return [float(x)]
    else:
        return list(x)


def skater_model_suffix(skater_name, k, n_input):
    return skater_name + '_' + str(k) + '_' + str(n_input) + '.json'


from tensorflow import keras

# TODO: Move into package shared by sklearned, maybe


def keras_optimizer_from_name(opt_name,learning_rate):
    if opt_name == 'SGD':
        return keras.optimizers.SGD(learning_rate=learning_rate)
    elif opt_name == 'RMSprop':
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif opt_name == 'Adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_name == 'Adagrad':
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif opt_name == 'Adadelta':
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif opt_name == 'Adamax':
        return  keras.optimizers.Adamax(learning_rate=learning_rate)
    elif opt_name == 'Nadam':
        return keras.optimizers.Nadam(learning_rate=learning_rate)
    elif opt_name =='Ftrl':
        return keras.optimizers.Ftrl(learning_rate=learning_rate)
    else:
        raise ValueError('Forgot '+opt_name)
