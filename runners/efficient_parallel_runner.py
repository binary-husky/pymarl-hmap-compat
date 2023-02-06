from UTIL.colorful import *
from UTIL.tensor_ops import np_one_hot
# from UTIL.tensor_ops import np_one_hot, __hash__, sychronize_experiment, __hashn__
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
from components.trajectory import OffPolicyTrajManager
import numpy as np
import copy


# class pymarl_compat_conn:
#     def __init__(self, rank) -> None:
#         self.rank = rank
#         # env = env_fn.x()

#         pass

#     def recv(self):
#         pass

#     def send(self, cmd_data):
#         cmd, data = cmd_data
#         if cmd == "step":
#             actions = data
#             # Take a step in the environment
#             reward, terminated, env_info = env.step(actions)
#             # Return the observations, avail_actions and state to make the next action
#             state = env.get_state()
#             avail_actions = env.get_avail_actions()
#             obs = env.get_obs()
#             remote.send({
#                 # Data for the next timestep needed to pick an action
#                 "state": state,
#                 "avail_actions": avail_actions,
#                 "obs": obs,
#                 # Rest of the data for the current timestep
#                 "reward": reward,
#                 "terminated": terminated,
#                 "info": env_info
#             })
#         elif cmd == "reset":
#             env.reset()
#             remote.send({
#                 "state": env.get_state(),
#                 "avail_actions": env.get_avail_actions(),
#                 "obs": env.get_obs()
#             })
#         elif cmd == "close":
#             env.close()
#             remote.close()
#             break
#         elif cmd == "get_env_info":
#             remote.send(env.get_env_info())
#         elif cmd == "get_stats":
#             remote.send(env.get_stats())
#         else:
#             raise NotImplementedError

class ParallelRunnerConf:
    use_eppr = False
    eppr_prob = 0.0025

class EfficientParallelRunner:

    def __init__(self, args, logger):
        print亮红('initializing ParallelRunner')
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        if 'env_uuid' in self.args.__dict__: self.args.env_args['env_uuid'] = self.args.__dict__['env_uuid']
        # Make subprocesses for the envs
        print亮红('initializing %d workers'%self.batch_size)
        # self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]

        self.hmp_remote_uuid = self.args.env_args['env_uuid']
        if platform.system() == "Windows":
            from UTIL.network import TcpClientP2P
            unix_path = ('localhost', args.compat_windows_port)
            self.remote_link_client = TcpClientP2P(unix_path, obj='pickle')
        else:
            unix_path = 'TEMP/Sockets/unix/%s'%self.hmp_remote_uuid
            from UTIL.network import UnixTcpClientP2P
            self.remote_link_client = UnixTcpClientP2P(unix_path, obj='pickle')

        self.env_info = self.remote_link_client.send_and_wait_reply(("get_env_info",))
        self.episode_limit = self.env_info["episode_limit"]

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self.traj_manager = OffPolicyTrajManager(n_env=self.batch_size, traj_limit=self.episode_limit, pool_size_limit=args.buffer_size)
        self.traj_manager.encountered_test_phase = False

    def setup(self, scheme, groups, preprocess, mac):
        # self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
        #                          preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        self.remote_link_client.__del__()
        return 
        # for parent_conn in self.parent_conns:
        #     parent_conn.send(("close", None))

    # remove terminated env info before stepping into next loop
    @staticmethod
    def pop_rm(ACTIVE, key):
        ter = ACTIVE["terminated"]
        content = ACTIVE.pop(key)
        assert len(ter) == len(content)
        return content[~ter]

    def 对齐线程字典更新(self, ALIGN, ACTIVE, 活跃线程ID):
        for k in ALIGN:
            ALIGN[k] = self.active2align(active_value=ACTIVE[k], active_ID=活跃线程ID, align_value=ALIGN[k]) \
                if hasattr(ACTIVE[k],'__getitem__') else ACTIVE[k]
        return ALIGN

    def active2align(self, active_value, active_ID, align_value, override=None):   # active_value: read only
        if override is not None: 
            align_value_ = [override]*len(align_value)
        else:
            align_value_ = copy.deepcopy(align_value)
        for i in range(len(align_value_)):
            if not isinstance(align_value_[i], bool) and align_value_[i] is not None: 
                align_value_[i] *= 0

        for 顺序ID, 线程ID in enumerate(active_ID): align_value_[线程ID] = active_value[顺序ID]
        return align_value_


    exclude_keys = ["time","state +1","avail_actions +1","obs +1","terminated","actions_onehot"]




    def run(self):
        # reset all env
        复位回馈 = self.remote_link_client.send_and_wait_reply(("reset_confirm_all",))
        # for parent_conn in self.parent_conns: parent_conn.send(("reset", None))
        # Get the obs, state and avail_actions back
        # 复位回馈 = [parent_conn.recv() for parent_conn in self.parent_conns]
        ACTIVE = {
            "state"             : np.array([d["state"] for d in 复位回馈]),
            "avail_actions"     : np.array([d["avail_actions"] for d in 复位回馈]),
            "obs"               : np.array([d["obs"] for d in 复位回馈]),
            "time"              : 0      };ACTIVE.update({
            "actions_onehot -1" : np.zeros_like(ACTIVE["avail_actions"]),
            "terminated"        : np.array([False for _ in range(self.batch_size)])
        })
        ALIGN = copy.deepcopy(ACTIVE)

        # 初始化两次，因为我想用  “活跃线程ID_时刻表[-2]”, call it with [-1] or [-2]
        活跃线程ID时刻表 = [list(range(self.batch_size)), list(range(self.batch_size)),]

        self.env_steps_this_run = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        # policy resonance
        if ParallelRunnerConf.use_eppr:
            pr_flag = (np.random.rand(self.batch_size) < ParallelRunnerConf.eppr_prob)
        else:
            pr_flag = None

        def deal_with_test_routine_when_necessary(步进回馈):
            entering_test_phase = (步进回馈 == 'entering_test_phase') or (步进回馈[0]['info']['testing'])
            # 反馈回来的是正常的数据，还是要求算法进入测试模式的数据, 如果是测试模式，则进入测试环境，直到测试模式结束
            if entering_test_phase: 
                self.test_run()
                步进回馈 = self.remote_link_client.send_and_wait_reply(("get_step_future", ))   # 重新读取，获得training环境的回馈
                entering_test_phase = (步进回馈 == 'entering_test_phase') or (步进回馈[0]['info']['testing'])
                assert not entering_test_phase
                self.traj_manager.encountered_test_phase = True
            return 步进回馈

        while True:
            ALIGN_TERM = np.array([False if i in 活跃线程ID时刻表[-1] else True for i in range(self.batch_size)])
            actions = self.mac.select_actions(ALIGN, t_env=self.t_env, bs=活跃线程ID时刻表[-1], test_mode=False, pr_flag=pr_flag)

            self.remote_link_client.send_and_wait_reply(
                ("step_all", self.active2align(actions, 活跃线程ID时刻表[-1], [None]*self.batch_size), ~ALIGN_TERM))
            步进回馈 = self.remote_link_client.send_and_wait_reply(("get_step_future", ))
            步进回馈 = deal_with_test_routine_when_necessary(步进回馈)

            self.env_steps_this_run += len(步进回馈)
            # 准备注入memory buffer pool
            ACTIVE.update({
                "actions"           : np.array(actions)         });ACTIVE.update({
                "rewards"           : np.array([d["reward"]         for d in 步进回馈]),
                "terminated"        : np.array([d["terminated"]     for d in 步进回馈]),
                "state +1"          : np.array([d["state"]          for d in 步进回馈]),
                "avail_actions +1"  : np.array([d["avail_actions"]  for d in 步进回馈]),
                "obs +1"            : np.array([d["obs"]            for d in 步进回馈])     }); ACTIVE.update({
                "actions_onehot"    : np_one_hot(ACTIVE["actions"], self.args.n_actions),
                "_SKIP_"            : np.array(ALIGN_TERM),
                "_DONE_"            : self.active2align(ACTIVE["terminated"], 活跃线程ID时刻表[-1], ALIGN_TERM),
                "_TOBS_"            : self.active2align(ACTIVE["obs +1"],     活跃线程ID时刻表[-1], ALIGN["obs"]), 
                "_TSTA_"            : self.active2align(ACTIVE["state +1"],   活跃线程ID时刻表[-1], ALIGN["state"]),
            })

            # 注入memory buffer pool
            self.traj_manager.feed_traj(ACTIVE, exclude=self.exclude_keys)

            # update 活跃线程ID时刻表
            活跃线程ID时刻表.append([env_id for ith, env_id in enumerate(活跃线程ID时刻表[-1]) if not 步进回馈[ith]["terminated"]])

            # For next mac.select_actions
            ACTIVE.update({
                "actions_onehot -1" : ACTIVE.pop("actions_onehot")[~ACTIVE["terminated"]]   });ACTIVE.update({
                "state"             : ACTIVE.pop("state +1")[~ACTIVE["terminated"]],
                "avail_actions"     : ACTIVE.pop("avail_actions +1")[~ACTIVE["terminated"]],
                "obs"               : ACTIVE.pop("obs +1")[~ACTIVE["terminated"]],
                "time"              : ACTIVE["time"] + 1,
            })
            ALIGN = self.对齐线程字典更新(ALIGN, ACTIVE, 活跃线程ID时刻表[-1])

            # 当所有线程都结束后，退出循环
            if all([d["terminated"] for d in 步进回馈]): break

        # 切除多余的轨迹，节省内存
        self.traj_manager.finalize_traj()
        self.t_env += self.env_steps_this_run
        self.mac.free_hidden_state()
        return self.traj_manager

    def test_run(self):
        clone_mac = copy.deepcopy(self.mac)
        batch_size = self.batch_size
        n_actions = self.args.n_actions
        while True:
            复位回馈 = self.remote_link_client.send_and_wait_reply(("reset_confirm_all",))
            ACTIVE = {
                "state"             : np.array([d["state"] for d in 复位回馈]),
                "avail_actions"     : np.array([d["avail_actions"] for d in 复位回馈]),
                "obs"               : np.array([d["obs"] for d in 复位回馈]),
                "time"              : 0      };ACTIVE.update({
                "actions_onehot -1" : np.zeros_like(ACTIVE["avail_actions"]),
                "terminated"        : np.array([False for _ in range(batch_size)])
            })
            ALIGN = copy.deepcopy(ACTIVE)
            活跃线程ID时刻表 = [list(range(batch_size)), list(range(batch_size)),]     
            clone_mac.init_hidden(batch_size=batch_size)
            while True:
                ALIGN_TERM = np.array([False if i in 活跃线程ID时刻表[-1] else True for i in range(self.batch_size)])
                actions = clone_mac.select_actions(ALIGN, t_env=self.t_env, bs=活跃线程ID时刻表[-1], test_mode=True)
                # Receive data back for each unterminated env
                self.remote_link_client.send_and_wait_reply(("step_all", self.active2align(actions, 活跃线程ID时刻表[-1], [None]*self.batch_size), ~ALIGN_TERM))
                步进回馈 = self.remote_link_client.send_and_wait_reply(("get_step_future", ))
                staying_test_phase = 步进回馈[0]['info']['testing']
                # 反馈回来的是正常的数据，还是要求算法进入测试模式的数据, 如果是测试模式，则进入测试环境，直到测试模式结束
                if not staying_test_phase: 
                    del clone_mac
                    return

                # 准备注入memory buffer pool
                ACTIVE.update({
                    "actions"           : np.array(actions)         });ACTIVE.update({
                    "rewards"           : np.array([d["reward"]         for d in 步进回馈]),
                    "terminated"        : np.array([d["terminated"]     for d in 步进回馈]),
                    "state +1"          : np.array([d["state"]          for d in 步进回馈]),
                    "avail_actions +1"  : np.array([d["avail_actions"]  for d in 步进回馈]),
                    "obs +1"            : np.array([d["obs"]            for d in 步进回馈])     }); ACTIVE.update({
                    "actions_onehot"    : np_one_hot(ACTIVE["actions"], n_actions),
                })
                # update 活跃线程ID时刻表
                活跃线程ID时刻表.append([env_id for ith, env_id in enumerate(活跃线程ID时刻表[-1]) if not 步进回馈[ith]["terminated"]])
                ACTIVE.update({
                    "actions_onehot -1" : ACTIVE.pop("actions_onehot")[~ACTIVE["terminated"]]   });ACTIVE.update({
                    "state"             : ACTIVE.pop("state +1")[~ACTIVE["terminated"]],
                    "avail_actions"     : ACTIVE.pop("avail_actions +1")[~ACTIVE["terminated"]],
                    "obs"               : ACTIVE.pop("obs +1")[~ACTIVE["terminated"]],
                    "time"              : ACTIVE["time"] + 1,
                })
                ALIGN = self.对齐线程字典更新(ALIGN, ACTIVE, 活跃线程ID时刻表[-1])
                if all([d["terminated"] for d in 步进回馈]): break