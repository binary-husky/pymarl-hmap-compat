import numpy as np
import pickle
import redis

class MultiAgentEnv(object):

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        raise NotImplementedError



def toRedis(self, redis_, array, key):
    self.redis_client = redis.Redis(host=self.sys_core_ip, port=6379)
    self.redis_client.set(key, pickle.dumps(array)) # 序列化列表并缓存到redis

def fromRedis(redis_, key):
    x = redis_.get(key)
    if x is None:
        return None
    return pickle.loads(x)

class HMP_compat(MultiAgentEnv):
    def __init__(self, map_name, **kwargs):
        import uuid
        # print(kwargs)
        # assert False
        # self.sub_client
        self.redis = redis.Redis(host='127.0.0.1', port=6379)
        self.parallel_uuid = uuid.uuid1().hex   # use uuid to identify threads
        self.remote_uuid = kwargs['env_uuid']
        pass

    def remote_call(self, cmd, args, uuid):
        # debug: flush Rx
        buf = self.redis.rpop('<<hmp%s'%self.parallel_uuid)
        if buf is not None:
            # print(buf)
            assert False, ('hmp <---> pymarl IO error!')
        # debug end

        cmd_arg = (cmd, args, uuid)    # tuple (cmd, args)
        self.redis.lpush('>>hmp%s'%self.remote_uuid, pickle.dumps(cmd_arg))
        _, buf = self.redis.brpop('<<hmp%s'%self.parallel_uuid)
        return pickle.loads(buf)

    def get_state_size(self):
        return self.remote_call(cmd="get_state_size", args=(), uuid=self.parallel_uuid)

    def get_current_mode(self):
        return self.remote_call(cmd="get_current_mode", args=(), uuid=self.parallel_uuid)

    def get_obs_size(self):
        return self.remote_call(cmd="get_obs_size", args=(), uuid=self.parallel_uuid)

    def get_total_actions(self):
        return self.remote_call(cmd="get_total_actions", args=(), uuid=self.parallel_uuid)

    def reset(self):
        assert self.remote_call(cmd="confirm_reset", args=(), uuid=self.parallel_uuid)

    def get_state(self):
        return self.remote_call(cmd="get_state_of", args=(), uuid=self.parallel_uuid)

    def get_avail_actions(self):
        return self.remote_call(cmd="get_avail_actions_of", args=(), uuid=self.parallel_uuid)

    def get_obs(self):
        return self.remote_call(cmd="get_obs_of", args=(), uuid=self.parallel_uuid)

    def step(self, actions):
        res = self.remote_call(cmd="step_of", args=(actions,), uuid=self.parallel_uuid)
        return res

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.remote_call(cmd="get_n_agents", args=(), uuid=self.parallel_uuid), #self.get_n_agents(), #self.n_agents,
                    "episode_limit": self.remote_call(cmd="get_episode_limit", args=(), uuid=self.parallel_uuid),}
        return env_info


    def get_stats(self):
        return self.remote_call(cmd="get_stats_of", args=(), uuid=self.parallel_uuid)

    def close(self):
        return self.remote_call(cmd="close", args=(), uuid=self.parallel_uuid)
