
# Cache ... should clean this up ... packaging is not working
import pathlib
import os
import json
from successor.conventions import skater_model_suffix, keras_optimizer_from_name
from tensorflow import keras
from successor.skaters.scalarskaters.remote import SKLEARNED_CHAMPIONS, get_remote_info, get_remote_model_spec, SKL_EXAMPLE_REMOTE_PATH
from pprint import pprint


ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent)
MODEL_CACHE = ROOT_PATH + os.path.sep + 'sklearnedmodelcache'
INFO_CACHE = ROOT_PATH + os.path.sep + 'sklearnedinfocache'


def local_info_path(skater_name, k: int, n_input: int):
    return INFO_CACHE + os.path.sep + skater_model_suffix(skater_name=skater_name, k=k, n_input=n_input)


def local_model_path(skater_name, k: int, n_input: int):
    return MODEL_CACHE + os.path.sep + skater_model_suffix(skater_name=skater_name,k=k,n_input=n_input)


def get_local_info(skater_name:str, k: int, n_input: int):
    with open(local_info_path(skater_name=skater_name, k=k, n_input=n_input),'rt') as fp:
        info = json.load(fp)
    return info





def save_model_locally(model: dict, skater_name: str, k: int, n_input: int):
    _file = local_model_path(skater_name=skater_name, k=k, n_input=n_input)
    with open(_file, 'wt') as fp:
        json.dump(obj=model,fp=fp)



def save_info_locally(info: dict, skater_name: str, k: int, n_input: int):
    _file = local_info_path(skater_name=skater_name, k=k, n_input=n_input)
    with open(_file, 'wt') as fp:
        json.dump(obj=info,fp=fp)



def copy_inventory_to_cache():
    for champ in SKLEARNED_CHAMPIONS:
        model = get_remote_model_spec(**champ)
        info  = get_remote_info(**champ)
        if model is not None and info is not None:
            save_model_locally(model=model,**champ)
            save_info_locally(info=info, **champ)



def get_local_model(skater_name:str, k: int, n_input: int):
    with open(local_model_path(skater_name=skater_name, k=k, n_input=n_input),'rt') as fp:
        model_json = fp.read()
    model = keras.models.model_from_json(model_json)
    return model

def get_local_compiled_model(skater_name:str,k:int,n_input:int):
    model = get_local_model(skater_name=skater_name,k=k,n_input=n_input)
    model_info = get_local_info(skater_name=skater_name,k=k,n_input=n_input)
    keras_optimizer = keras_optimizer_from_name(opt_name=model_info['keras_optimizer'], learning_rate=model_info['learning_rate'])
    model.compile(optimizer=keras_optimizer)
    return model



if __name__ == '__main__':
    print('example model location is ' + SKL_EXAMPLE_REMOTE_PATH)
    copy_inventory_to_cache()
    n_input = 160
    model = get_local_compiled_model(skater_name='tsa_aggressive_d0_ensemble',k=1,n_input=n_input)
    pprint(model)
    import numpy as np
    x = np.random.randn(300,1,n_input)
    y = model.predict(x)
    pass