# cython: language_level=3
import numpy as np
import copy
from .traj import TRAJ_BASE
from UTIL.colorful import *
from UTIL.tensor_ops import __hash__, my_view, np_one_hot, np_repeat_at, np_softmax, stack_padding
def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class trajectory(TRAJ_BASE):

    def __init__(self, traj_limit, env_id):
        super().__init__(traj_limit, env_id)
        self.reference_track_name = 'state'

    def early_finalize(self):
        assert not self.readonly_lock   # unfinished traj
        self.need_reward_bootstrap = True

    def set_terminal_obs(self, tobs, tstate=None):
        self.tobs = copy.deepcopy(tobs)
        self.tstate = copy.deepcopy(tstate)

    def reward_push_forward(self, dead_mask):
        for i in reversed(range(self.time_pointer)):
            if i==0: continue
            self.reward[i-1] += self.reward[i]* dead_mask[i].astype(np.int)
            self.reward[i] = self.reward[i]* (~dead_mask[i]).astype(np.int)

    # new finalize
    def finalize(self):
        self.readonly_lock = True
        assert not self.deprecated_flag
        TJ = lambda key: getattr(self, key) 
        # deadmask
        dead_mask = (np.isnan(my_view(self.obs, [0,0,-1]))).all(-1)
        self.reward_push_forward(dead_mask) # push terminal reward forward
        threat = np.zeros(shape=dead_mask.shape) - 1
        assert dead_mask.shape[0] == self.time_pointer
        for i in reversed(range(self.time_pointer)):
            # threat[:(i+1)] 不包含threat[(i+1)]
            if i+1 < self.time_pointer:
                threat[:(i+1)] += (~(dead_mask[i+1]&dead_mask[i])).astype(np.int)
            elif i+1 == self.time_pointer:
                threat[:] += (~dead_mask[i]).astype(np.int)
        
        '''
        filter = None
        s_time = threat[0] + 1
        if np.percentile(s_time, 50) < 100:
            # 即训练初期，大部分智能体不能活到100步
            new_reward = TJ('reward')
            filter = s_time > (np.mean(s_time) + np.std(s_time)*1)
            exception = (s_time < self.traj_limit) & (new_reward[threat==0] >=0)
            filter = filter & (~exception) 
            # 苟活时间大于整体存活时间 + sigma的个体加以惩罚
            penelty = -(filter).astype(np.float)*1.0
            # print(new_reward[threat==0])
            new_reward[threat==0] = new_reward[threat==0] + penelty
            # print(new_reward[threat==0])
            setattr(self, 'reward', new_reward)
            
        '''
            
        SAFE_LIMIT = 11
        threat = np.clip(threat, -1, SAFE_LIMIT)
        setattr(self, 'threat', np.expand_dims(threat, -1))


        # ! Use GAE to calculate return
        self.gae_finalize_return(reward_key='reward', value_key='value', new_return_name='return')
        return

    # def gae_finalize_return(self, reward_key, value_key, new_return_name):
    #     # ------- gae parameters -------
    #     gamma = AlgorithmConfig.gamma 
    #     tau = AlgorithmConfig.tau
    #     # ------- -------------- -------
    #     rewards = getattr(self, reward_key)
    #     value = getattr(self, value_key)
    #     length = rewards.shape[0]
    #     assert rewards.shape[0]==value.shape[0]
    #     # if dimension not aligned
    #     if rewards.ndim == value.ndim-1: rewards = np.expand_dims(rewards, -1)
    #     # initalize two more tracks
    #     setattr(self, new_return_name, np.zeros_like(value))
    #     self.key_dict.append(new_return_name)

    #     returns = getattr(self, new_return_name)
    #     boot_strap = 0 if not self.need_reward_bootstrap else self.boot_strap_value['bootstrap_'+value_key]

    #     for step in reversed(range(length)):
    #         if step==(length-1): # 最后一帧
    #             value_preds_delta = rewards[step] + gamma * boot_strap      - value[step]
    #             gae = value_preds_delta
    #         else:
    #             value_preds_delta = rewards[step] + gamma * value[step + 1] - value[step]
    #             gae = value_preds_delta + gamma * tau * gae
    #         returns[step] = gae + value[step]














class TrajPoolManager(object):
    def __init__(self, n_pool):
        self.n_pool =  n_pool
        self.cnt = 0

    def absorb_finalize_pool(self, pool):
        for traj_handle in pool:
            traj_handle.cut_tail()
        pool = list(filter(lambda traj: not traj.deprecated_flag, pool))
        for traj_handle in pool: traj_handle.finalize()
        self.cnt += 1
        task = ['train']
        return task, pool





'''
    轨迹池管理
'''

class TrajManagerBase(object):
    def __init__(self, n_env, traj_limit):
        self.n_env = n_env
        self.traj_limit = traj_limit
        self.update_cnt = 0
        self.traj_pool = []
        self.registered_keys = []
        self.live_trajs = [trajectory(self.traj_limit, env_id=i) for i in range(self.n_env)]
        self.live_traj_frame = [0 for _ in range(self.n_env)]
        self._traj_lock_buf = None
        self.patience = 1000
        pass
    
    def __check_integraty(self, traj_frag):
        if self.patience < 0: return # stop wasting time checking this
        self.patience -= 1
        for key in traj_frag:
            if key not in self.registered_keys and (not key.startswith('_')):
                self.registered_keys.append(key)
        for key in self.registered_keys:
            assert key in traj_frag, ('this key sometimes disappears from the traj_frag:', key)

    def batch_update(self, traj_frag):
        self.__check_integraty(traj_frag)
        done = traj_frag.pop('_DONE_') # done flag
        skip = traj_frag.pop('_SKIP_') # skip/frozen flag
        tobs = traj_frag.pop('_TOBS_') # terminal obs
        tsta = traj_frag.pop('_TSTA_') if '_TSTA_' in traj_frag else [None for _ in tobs]
        # single bool to list bool
        if isinstance(done, bool): done = [done for i in range(self.n_env)]
        if isinstance(skip, bool): skip = [skip for i in range(self.n_env)]
        n_active = sum(~skip)
        # feed
        cnt = 0
        for env_i in range(self.n_env):
            if skip[env_i]: continue
            # otherwise
            frag_index = cnt; cnt += 1
            env_index = env_i
            traj_handle = self.live_trajs[env_index]
            for key in traj_frag:
                self.traj_remember(traj_handle, key=key, content=traj_frag[key],frag_index=frag_index, n_active=n_active)
            self.live_traj_frame[env_index] += 1
            traj_handle.time_shift()
            if done[env_i]:
                assert tobs[env_i] is not None # get the final obs
                traj_handle.set_terminal_obs(tobs[env_i], tsta[env_i])
                self.traj_pool_append(traj_handle)
                self.live_trajs[env_index] = trajectory(self.traj_limit, env_id=env_index)
                self.live_traj_frame[env_index] = 0




    def traj_remember(self, traj, key, content, frag_index, n_active):
        if content is None: traj.remember(key, None)
        elif isinstance(content, dict):
            for sub_key in content: 
                self.traj_remember(traj, "".join((key , ">" , sub_key)), content=content[sub_key], frag_index=frag_index, n_active=n_active)
        else:
            assert n_active == len(content), ('length error')
            traj.remember(key, content[frag_index]) # *


class BatchTrajManager(TrajManagerBase):
    def __init__(self, n_env, traj_limit, trainer_hook):
        super().__init__(n_env, traj_limit)
        self.trainer_hook = trainer_hook
        self.traj_limit = traj_limit
        self.train_traj_needed = AlgorithmConfig.train_traj_needed
        self.upper_training_epoch = AlgorithmConfig.upper_training_epoch
        self.pool_manager = TrajPoolManager(n_pool=self.upper_training_epoch)

    def update(self, traj_frag, index):
        assert traj_frag is not None
        for j, env_i in enumerate(index):
            traj_handle = self.live_trajs[env_i]
            for key in traj_frag:
                if traj_frag[key] is None:
                    assert False, key
                if isinstance(traj_frag[key], dict):  # 如果是二重字典，特殊处理
                    for sub_key in traj_frag[key]:
                        content = traj_frag[key][sub_key][j]
                        traj_handle.remember(key + ">" + sub_key, content)
                else:
                    content = traj_frag[key][j]
                    traj_handle.remember(key, content)
            self.live_traj_frame[env_i] += 1
            traj_handle.time_shift()
        return

    # 函数入口
    def feed_traj(self, traj_frag, require_hook=False):
        # an unlock hook must be executed before new trajectory feed in
        assert self._traj_lock_buf is None
        if require_hook: 
            # the traj_frag is not intact, lock up traj_frag, wait for more
            assert '_SKIP_' in traj_frag
            assert '_DONE_' not in traj_frag
            assert 'reward' not in traj_frag
            self._traj_lock_buf = traj_frag
            return self.unlock_fn
        else:
            assert '_DONE_' in traj_frag
            assert '_SKIP_' in traj_frag
            self.batch_update(traj_frag=traj_frag)
            return

        
    def train_and_clear_traj_pool(self):
        print('do update %d'%self.update_cnt)

        current_task_l, self.traj_pool = self.pool_manager.absorb_finalize_pool(pool=self.traj_pool)
        for current_task in current_task_l:
            ppo_update_cnt = self.trainer_hook(self.traj_pool, current_task)

        self.traj_pool = []
        self.update_cnt += 1
        # assert ppo_update_cnt == self.update_cnt
        return self.update_cnt




from config import GlobalConfig

class OffPolicyTrajManager(TrajManagerBase):
    def __init__(self, n_env, traj_limit, pool_size_limit):
        super().__init__(n_env, traj_limit)
        self.pool_size_limit = pool_size_limit
        self.traj_limit = traj_limit
        # self.train_traj_needed = AlgorithmConfig.train_traj_needed
        # self.upper_training_epoch = AlgorithmConfig.upper_training_epoch
        # self.pool_manager = TrajPoolManager(n_pool=self.upper_training_epoch)

    def update(self, traj_frag, index):
        assert traj_frag is not None
        for j, env_i in enumerate(index):
            traj_handle = self.live_trajs[env_i]
            for key in traj_frag:
                if traj_frag[key] is None:
                    assert False, key
                if isinstance(traj_frag[key], dict):  # 如果是二重字典，特殊处理
                    for sub_key in traj_frag[key]:
                        content = traj_frag[key][sub_key][j]
                        traj_handle.remember(key + ">" + sub_key, content)
                else:
                    content = traj_frag[key][j]
                    traj_handle.remember(key, content)
            self.live_traj_frame[env_i] += 1
            traj_handle.time_shift()
        return

    # 函数入口
    def feed_traj(self, traj_frag, exclude=[], require_hook=False):
        # an unlock hook must be executed before new trajectory feed in
        assert self._traj_lock_buf is None
        assert '_DONE_' in traj_frag
        assert '_SKIP_' in traj_frag
        traj_frag_exclude = copy.copy(traj_frag)
        for i_exclude in exclude: traj_frag_exclude.pop(i_exclude)
        self.batch_update(traj_frag=traj_frag_exclude)
        return

    def finalize_traj(self):
        for traj_handle in self.traj_pool:
            traj_handle.cut_tail()
        self.traj_pool = list(filter(lambda traj: not traj.deprecated_flag, self.traj_pool))
        
        return

    def train_and_clear_traj_pool(self):
        print('do update %d'%self.update_cnt)

        current_task_l, self.traj_pool = self.pool_manager.absorb_finalize_pool(pool=self.traj_pool)
        for current_task in current_task_l:
            ppo_update_cnt = self.trainer_hook(self.traj_pool, current_task)

        self.traj_pool = []
        self.update_cnt += 1
        return self.update_cnt

    def traj_pool_append(self, traj_handle):
        self.traj_pool.append(traj_handle)
        if len(self.traj_pool)>self.pool_size_limit:
            self.traj_pool.pop(0)

    def can_sample(self, train_traj_needed):
        if GlobalConfig.activate_logger: GlobalConfig.mcv.rec(len(self.traj_pool), 'PoolSize') 

        if len(self.traj_pool) >= train_traj_needed:  return True
        else:  return False

    def sample_pool_subset(self, train_traj_needed):
        # print('调试模式，无随机')
        # return self.traj_pool
        n_episodes_in_buffer = len(self.traj_pool)
        ep_ids = np.random.choice(n_episodes_in_buffer, train_traj_needed, replace=False)
        traj_pool_subset = [self.traj_pool[ep_id] for ep_id in ep_ids]
        return traj_pool_subset

    def parse(self, traj_pool, key_name, method, padding=np.nan):
        if method == 'normal':
            # [0,1,2,3,4] --> [0,1,2,3,4] 
            set_item = stack_padding([getattr(traj, key_name) for traj in traj_pool], padding)

        elif '+$' in method:
            # [0,1,2,3,4] --> [0,1,2,3,4,traj.tail_key] 
            _, tail_key = method.split('$')
            stack_buff = [
                np.concatenate( (
                    getattr(traj, key_name),
                    np.expand_dims(getattr(traj, tail_key), axis=0)
                ), axis=0) for traj in traj_pool ]
            set_item = stack_padding(stack_buff, padding)

        elif '+dupl':
            # [0,1,2,3,4] --> [0,1,2,3,4,4] 
            stack_buff = []
            for traj in traj_pool:
                key_traj = getattr(traj, key_name)
                key_traj = np.concatenate( (key_traj, key_traj[-1:]), axis=0)
                stack_buff.append(key_traj)
            set_item = stack_padding(stack_buff, padding)

        elif '+zero':
            # [0,1,2,3,4] --> [0,1,2,3,4,4] 
            stack_buff = []
            for traj in traj_pool:
                key_traj = getattr(traj, key_name)
                key_traj = np.concatenate( (key_traj, key_traj[-1:]*0), axis=0)
                stack_buff.append(key_traj)
            set_item = stack_padding(stack_buff, padding)

        elif '+nan':
            # [0,1,2,3,4] --> [0,1,2,3,4,4] 
            stack_buff = []
            for traj in traj_pool:
                key_traj = getattr(traj, key_name)
                key_traj = np.concatenate( (key_traj, key_traj[-1:]+np.nan), axis=0)
                stack_buff.append(key_traj)
            set_item = stack_padding(stack_buff, padding)

        else:
            assert False
        # np.concatenate([getattr(traj, key_name) for traj in traj_pool], axis=0)
        # [getattr(traj, key_name).shape for traj in traj_pool]
        return set_item