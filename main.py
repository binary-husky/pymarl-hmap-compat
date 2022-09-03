import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import json
import torch as th
from utils.logging import get_logger
import yaml
from UTIL.colorful import *
from run import REGISTRY as run_REGISTRY
from config_args import load_config_via_json

# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment("pymarl",save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds
results_path = join(dirname(dirname(abspath(__file__))), "results")


def 解密字符串(p):
    k = ''.join(['@']*1000)
    dec_str = ""
    for i,j in zip(p.split("_")[:-1],k):
        # i 为加密字符，j为秘钥字符
        temp = chr(int(i) - ord(j))
        dec_str = dec_str+temp
    return dec_str

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    # 不知pymarl用的什么操作，非要把参数加密了，才能完整地传输过来
    pymarl_config_injection = 解密字符串(config['pymarl_config_injection'])
    load_config_via_json(json.loads(pymarl_config_injection), vb=True)

    from config import GlobalConfig
    config['batch_size_run'] = GlobalConfig.n_thread
    config['batch_size'] = GlobalConfig.batch_size
    config['t_max'] = GlobalConfig.t_max

    np.random.seed(GlobalConfig.seed)
    th.manual_seed(GlobalConfig.seed)
    config['env_args']['seed'] = GlobalConfig.seed
    if GlobalConfig.activate_logger:
        from VISUALIZE.mcom import mcom
        GlobalConfig.mcv = mcom(ip='127.0.0.1',
                    port=12086,
                    path='%s/%s/PymarlLog/'%(GlobalConfig.HmpRoot, GlobalConfig.logdir),
                    digit=16,
                    rapid_flush=True,
                    draw_mode=GlobalConfig.draw_mode,
                    tag='[pymarl-main.py]')
    # run
    run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                print亮靛('loading configuration from file:', f.name)
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            print亮靛('loading configuration from file:', f.name)
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name']) 
    # file_obs_path = join(results_path, "sacred", map_name, algo_name)
    
    # logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    # ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
