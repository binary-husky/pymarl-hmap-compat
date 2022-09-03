from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from UTIL.tensor_ops import Args2tensor, Args2tensor_Return2numpy
from UTIL.tensor_ops import np_one_hot, __hash__
# This multi-agent controller shares parameters between agents

class my_NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(my_NMAC, self).__init__(scheme, groups, args)

    @Args2tensor_Return2numpy
    def select_actions(self, frame_data, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = frame_data["avail_actions"]
        qvals = self.forward_(frame_data, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward_(self, frame_data, test_mode=False):
        agent_inputs = self._build_inputs(frame_data, frame_data["time"])
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # (8 env, 6 agent, 30 core dim)
        # self.hidden_states 8, 6, 64
        return agent_outs

    @Args2tensor    #_Return2numpy
    def forward(self, frame_data, test_mode=False):
        agent_inputs = self._build_inputs(frame_data, frame_data["time"])
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs
