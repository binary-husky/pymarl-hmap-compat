import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from UTIL.colorful import *
from utils.logging import get_mcv_logger


def main_loop(runner):
    # Run for a whole episode at a time
    print亮紫('Run for a whole episode at a time')
    mcv = get_mcv_logger()
    mcv.rec(runner.mac.action_selector.schedule.eval(runner.t_env), 'epsilon') 

    print亮绿(
        'Env counters:', 
        ' episode:',episode,
        ' runner.t_env:',runner.t_env,
        ' args.test_interval:', args.test_interval,
        ' last_test_T:', last_test_T,
        ' args.t_max:', args.t_max,
        ' args.log_interval:', args.log_interval,
        ' args.batch_size:', args.batch_size,   # 这个batch size指的是每次训练中使用的episode的数量
        ' args.batch_size_run:', args.batch_size_run    # 这个batch size指的是并行的环境（进程）的数量
    )
    # ? eval, collect episode data with runner.run
    with th.no_grad():
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

    if buffer.can_sample(args.batch_size):
        episode_sample = buffer.sample(args.batch_size)

        # Truncate batch to only filled timesteps
        max_ep_t = episode_sample.time_of_longest_episode()
        episode_sample = episode_sample[:, :max_ep_t]

        if episode_sample.device != args.device:
            episode_sample.to(args.device)

        learner.train(episode_sample, runner.t_env, episode)
        del episode_sample

    # Execute test runs once in a while
    n_test_runs = max(1, args.test_nepisode // runner.batch_size)
    if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
        # run test run 
        logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
        logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
            time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
        last_time = time.time()

        last_test_T = runner.t_env
        for _ in range(n_test_runs):
            runner.run(test_mode=True) # 结果保存到runner.batch

    if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
        model_save_time = runner.t_env
        save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
        #"results/models/{}".format(unique_token)
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Saving models to {}".format(save_path))

        # learner should handle saving/loading -- delegate actor save/load to mac,
        # use appropriate filenames to do critics, optimizer states
        learner.save_models(save_path)

    episode += args.batch_size_run

    if (runner.t_env - last_log_T) >= args.log_interval:
        logger.log_stat("episode", episode, runner.t_env)
        logger.print_recent_stats()
        last_log_T = runner.t_env

# start training
episode = 0
last_test_T = -args.test_interval - 1
last_log_T = 0
model_save_time = 0

start_time = time.time()
last_time = start_time

logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
while runner.t_env <= args.t_max:
    main_loop(runner)