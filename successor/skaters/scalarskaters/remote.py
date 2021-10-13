import pathlib
import os
from getjson import getjson
from pprint import pprint
import json
from successor.skaters.scalarskaters.sklearnedinventory import SKLEARNED_CHAMPION_URLS
from tensorflow import keras
from successor.conventions import keras_optimizer_from_name, skater_model_suffix



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


def remote_info_path(skater_name, k: int, n_input: int):
    return SKL_REMOTE_INFO_PATH + '/' + skater_model_suffix(skater_name=skater_name,k=k,n_input=n_input)


def remote_model_path(skater_name, k: int, n_input: int):
    return SKL_REMOTE_MODEL_PATH + '/' + skater_model_suffix(skater_name=skater_name, k=k, n_input=n_input)


def get_remote_model_spec(skater_name:str, k: int, n_input: int):
    _path = remote_model_path(skater_name=skater_name, k=k, n_input=n_input)
    _res = getjson(_path)
    if isinstance(_res,str):
        return json.loads(_res)
    elif isinstance(_res,dict):
        return _res
    else:
        print('Missing '+remote_model_path(skater_name=skater_name,k=k,n_input=n_input))


def get_remote_info(skater_name:str, k: int, n_input: int):
    _path = remote_info_path(skater_name=skater_name, k=k, n_input=n_input)
    _res = getjson(_path)
    if isinstance(_res,str):
        return json.loads(_res)
    elif isinstance(_res,dict):
        return _res
    else:
        print('Missing '+remote_info_path(skater_name=skater_name,k=k,n_input=n_input))


def get_remote_model(skater_name:str, k: int, n_input: int):
    _path = remote_model_path(skater_name=skater_name, k=k, n_input=n_input)
    _res = getjson(_path)
    if isinstance(_res,dict):
        _res = json.dumps(_res)
    model = keras.models.model_from_json(_res)
    return model


def get_remote_compiled_model(skater_name:str,k:int,n_input:int):
    model = get_remote_model(skater_name=skater_name, k=k, n_input=n_input)
    model_info = get_remote_info(skater_name=skater_name, k=k, n_input=n_input)
    keras_optimizer = keras_optimizer_from_name(opt_name=model_info['keras_optimizer'], learning_rate=model_info['learning_rate'])
    model.compile(optimizer=keras_optimizer)
    return model


