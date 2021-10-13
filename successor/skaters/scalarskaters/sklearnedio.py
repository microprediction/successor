import pathlib
import os
from getjson import getjson
from pprint import pprint
import json
from successor.skaters.scalarskaters.sklearnedinventory import SKLEARNED_CHAMPION_URLS
from tensorflow import keras
from successor.conventions import keras_optimizer_from_name

ROOT_PATH = str(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent)
MODEL_CACHE = ROOT_PATH + os.path.sep + 'sklearnedmodelcache'
INFO_CACHE = ROOT_PATH + os.path.sep + 'sklearnedinfocache'
SKL_REMOTE_PATH = 'https://raw.githubusercontent.com/microprediction/sklearned/main'
SKL_REMOTE_MODEL_PATH = SKL_REMOTE_PATH + '/champion_models'
SKL_REMOTE_INFO_PATH = SKL_REMOTE_PATH + '/champion_info'
SKL_EXAMPLE_REMOTE_PATH = SKL_REMOTE_MODEL_PATH + os.path.sep + 'tsa_aggressive_combined_ensemble_1_80.json'


def champion_from_url(url):
    # e.g. https: // raw.githubusercontent.com / microprediction / sklearned / main / champion_models / tsa_aggressive_combined_ensemble_1_80.json
    last_bit = url.split('/')[-1].replace('.json','')
    n_input = int(last_bit.split('_')[-1])
    k = int(last_bit.split('_')[-2])
    skater_name = '_'.join( last_bit.split('_')[:-2])
    return dict(skater_name=skater_name,k=k,n_input=n_input)

SKLEARNED_CHAMPIONS = [champion_from_url(url) for url in SKLEARNED_CHAMPION_URLS]


def champion_json(skater_name,k,n_input):
    return skater_name + '_' + str(k) + '_' + str(n_input) + '.json'


def remote_info_path(skater_name, k: int, n_input: int):
    return SKL_REMOTE_INFO_PATH + '/' + champion_json(skater_name=skater_name,k=k,n_input=n_input)


def remote_model_path(skater_name, k: int, n_input: int):
    return SKL_REMOTE_MODEL_PATH + '/' + champion_json(skater_name=skater_name, k=k, n_input=n_input)


def local_model_path(skater_name, k: int, n_input: int):
    return MODEL_CACHE + os.path.sep + champion_json(skater_name=skater_name,k=k,n_input=n_input)


def local_info_path(skater_name, k: int, n_input: int):
    return INFO_CACHE + os.path.sep + champion_json(skater_name=skater_name,k=k,n_input=n_input)



def _get_remote_model(skater_name:str, k: int, n_input: int):
    _path = remote_model_path(skater_name=skater_name, k=k, n_input=n_input)
    _res = getjson(_path)
    if isinstance(_res,str):
        return json.loads(_res)
    elif isinstance(_res,dict):
        return _res
    else:
        print('Missing '+remote_model_path(skater_name=skater_name,k=k,n_input=n_input))


def _get_remote_info(skater_name:str, k: int, n_input: int):
    _path = remote_info_path(skater_name=skater_name, k=k, n_input=n_input)
    _res = getjson(_path)
    if isinstance(_res,str):
        return json.loads(_res)
    elif isinstance(_res,dict):
        return _res
    else:
        print('Missing '+remote_info_path(skater_name=skater_name,k=k,n_input=n_input))


def get_local_model(skater_name:str, k: int, n_input: int):
    with open(local_model_path(skater_name=skater_name, k=k, n_input=n_input),'rt') as fp:
        model_json = fp.read()
    model = keras.models.model_from_json(model_json)
    return model

def get_local_info(skater_name:str, k: int, n_input: int):
    with open(local_info_path(skater_name=skater_name, k=k, n_input=n_input),'rt') as fp:
        model = json.load(fp)
    return model


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
        model = _get_remote_model(**champ)
        info  = _get_remote_info(**champ)
        if model is not None and info is not None:
            save_model_locally(model=model,**champ)
            save_info_locally(info=info, **champ)


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
