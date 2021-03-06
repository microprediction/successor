from momentum.skatertools.parade import parade
from successor.extension.reflection import reflect
from successor.skaters.scalarskaters.remote import get_remote_tensorflow
from successor.interpolation.linear import linear_interpolator
from successor.conventions import wrap, nonecast
import numpy as np
import time
from successor.skaters.scalarskaters.dimensional import squeeze_out_middle



def scaler_skater_factory(y, s, k:int, skater_name:str, n_input:int, extender=None, interpolator=None, local=False):
    """ Runs keras surrogate models, extending the time-series as required.

            skater_name
            n_input
            extender    : Takes [ float ] len<n_input possibly, to array (1, 1, n_input)
            local

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
                if local:
                    raise NotImplementedError('Won''t work at present')
                else:
                    s['models'][model_k] = get_remote_tensorflow(skater_name=skater_name, k=model_k, n_input=n_input, verbose=False)
                n_models += 1
            except:
                print('No surrogate found for k='+str(model_k)+' so interpolation or extrapolation will be used.')
        if n_models==0:
            raise LookupError('Cannot instantiate as no model was found for '+skater_name)
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
        # Rescale
        scale_factor = np.mean([ abs(y_) for y_ in y_extended ])+1

        y_extended_input = np.ndarray(shape=(1,1,n_input))
        y_extended_input[0,0,:] = [ y_/scale_factor for y_ in y_extended]
        x_with_nan = [np.nan for _ in range(k)]
        for model_k, k_model in s['models'].items():
            x_ = scale_factor*k_model(y_extended_input) # faster then k_model.predict( )
                                # https://github.com/tensorflow/tensorflow/commit/42f469be0f3e8c36624f0b01c571e7ed15f75faf
            xq_ = squeeze_out_middle(x_)
            x_with_nan[model_k-1] = float(xq_[0,0])

        # Interpolate
        x = interpolator(x_with_nan)

        # Use empirical x_std, ignore empirical bias
        _x_bias, x_std, s['p'] = parade(p=s['p'], x=x, y=y0)
        x_std_fallback = nonecast(x_std,fill_value=1.0)

        s['cpu']['invocation']+= time.time()-invocation_start_time
        s['cpu']['count']+=1
        return x, x_std_fallback, s







