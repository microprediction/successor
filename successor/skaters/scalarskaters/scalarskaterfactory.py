from momentum.skatertools.parade import parade
from successor.extension.reflection import reflect
from successor.skaters.scalarskaters.sklearnedio import get_local_compiled_model
from successor.interpolation.linear import linear_interpolator
from successor.conventions import wrap, nonecast
import numpy as np
import time



def scaler_skater_factory(y, s, k:int, skater_name:str, n_input:int, extender=None, interpolator=None):
    """ Runs keras surrogate models, extending the time-series as required.

            skater_name
            n_input
            extender    : Takes [ float ] len<n_input possibly, to array (1, 1, n_input)

        This will try to find local models for each k. If it cannot, it will interpolate or extrapolate based on the
        surrogates that it is able to find. By default interpolation is linear and extrapolation is nearest neighbour.

    """
    if extender is None:
        extender = reflect

    if interpolator is None:
        interpolator = linear_interpolator

    y0 = wrap(y)[0]
    if not s.get('p'):
        init_start_time = time.time()
        s = {'p':{},
             'models':dict(),
             'y':[]}
        n_models = 0
        for model_k in range(1,k+1):
            try:
                s['models'][model_k] = get_local_compiled_model(skater_name=skater_name,k=model_k,n_input=n_input)
                n_models += 1
            except:
                print('No surrogate found for k='+str(k)+' so interpolation or extrapolation will be used.')
        if n_models==0:
            raise LookupError('Cannot instantiate as no local model was found for '+skater_name)
        s['cpu'] = {'initialization':time.time()-init_start_time,'invocation':0,'count':0}
    if y0 is None:
        return None, s, None
    else:
        invocation_start_time = time.time()
        s['y'].append(y0)

        # Use neural networks to predict
        y_extended = extender(s['y'][:n_input],n_input=n_input)
        if not len(y_extended)==n_input:
            raise IndexError('wrong length returned by extender')
        y_extended_input = np.ndarray(shape=(1,1,n_input))
        y_extended_input[0,0,:] = y_extended
        x_hat = [np.nan for _ in range(k)]
        for model_k, k_model in s['models'].items():
            x = k_model(y_extended_input) # faster then k_model.predict( )
                                # https://github.com/tensorflow/tensorflow/commit/42f469be0f3e8c36624f0b01c571e7ed15f75faf
            x_hat[model_k-1] = x[0,0,0]

        # Interpolate
        x = interpolator(x_hat)

        # Use empirical x_std, ignore empirical bias
        _x_bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)
        x_std_fallback = nonecast(x_std,fill_value=1.0)

        s['cpu']['invocation']+= time.time()-invocation_start_time
        s['cpu']['count']+=1
        return x, x_std_fallback, s







