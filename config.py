import time
import numpy as np
from UTIL.colorful import *

class ChainVar(object):
    def __init__(self, chain_func, chained_with):
        self.chain_func = chain_func
        self.chained_with = chained_with

class GlobalConfig(object): # ADD_TO_CONF_SYSTEM //DO NOT remove this comment//
    runner = 'efficient_parallel_runner'
    draw_mode = 'Img'  # 'Web','Native'
    logdir = './ZHECKPOINT/test/'
    cfg_ready = True
    device = 'cuda'
    batch_size = 128
    n_thread = 8
    use_float64 = False
    load_checkpoint = False
    seed = 0
    manual_gpu_ctl = False
    gpu_fraction = 1
    HmpRoot = ''
    ExpNote = ''
    activate_logger = True

    train_time_testing = True
    test_interval = 128
    test_only = False
    test_epoch = 32
    
    t_max = 1e12    # was 1e7, big enough for hmp to take full control

    compat_windows_port = 12235
