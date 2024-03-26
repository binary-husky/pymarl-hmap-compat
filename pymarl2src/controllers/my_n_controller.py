from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch
from utils.rl_utils import RunningMeanStd
import numpy as np
from UTIL.tensor_ops import Args2tensor, Args2tensor_Return2numpy
from UTIL.tensor_ops import np_one_hot, __hash__
from commom.vae import VanillaVAE
# This multi-agent controller shares parameters between agents

# WUWE design principle: where you use it, where you define it.
class PymarlAlgorithmConfig:
    use_normalization = False
    use_vae = False
    hidden_shape = 64
    encoded_dim = 32
    agent_rnn_hidden_dim = 64
    kld_loss_weight = 1
    recons_loss_weight = 1

    DEBUG_DETACH_INPUTS = False
    vae_lr_ratio = 2.0

class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = input_shape = self._get_input_shape(scheme)
        self.use_vae = PymarlAlgorithmConfig.use_vae
        if self.use_vae:
            self._build_agents(PymarlAlgorithmConfig.encoded_dim)
        else:
            self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)
        self.hidden_states = None

        self.use_normalization = PymarlAlgorithmConfig.use_normalization
        if self.use_normalization:
            self._batch_norm = None
            # 想在这里初始化归一模块，但是不行，因为有两个mac需要共享归一模块
            # self._batch_norm = DynamicNorm(input_shape, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
        if self.use_vae:
            self.vae_module = VanillaVAE(input_dim=input_shape, latent_dim=PymarlAlgorithmConfig.encoded_dim, 
                hidden_dim=PymarlAlgorithmConfig.hidden_shape, degenerate2ae=True,)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def free_hidden_state(self):
        self.hidden_states = None

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        if self.use_vae:
            return [self.agent.parameters(), self.vae_module.parameters() ]
        else:
            return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        if self.use_vae:
            self.vae_module.cuda()

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))
        if self.use_vae:
            torch.save(self.vae_module.state_dict(), "{}/vae_module.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.use_vae:
            self.vae_module.load_state_dict(torch.load("{}/vae_module.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        from modules.agents.n_rnn_agent import NRNNAgent
        self.agent = NRNNAgent(input_shape, PymarlAlgorithmConfig.agent_rnn_hidden_dim, self.args.n_actions)

    def _build_inputs(self, batch, t, get_vae_loss=False, valid_mask=None):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = len(batch["obs"]) # batch.batch_size
        inputs = []
        inputs.append(batch["obs"])  # b1av
        if self.args.obs_last_action: # 将上一时刻的动作作为此刻的观测
            inputs.append(batch["actions_onehot -1"])
        if self.args.obs_agent_id:  # 添加agent的编号作为输入
            inputs.append(torch.eye(self.n_agents, device=batch["obs"].device).unsqueeze(0).expand(bs, -1, -1))

        inputs = torch.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        if PymarlAlgorithmConfig.DEBUG_DETACH_INPUTS: 
            inputs = torch.zeros_like(inputs)
        if self.use_normalization:
            inputs = self._batch_norm(inputs)
            # print('Good!')

        if self.use_vae:
            if get_vae_loss:
                # 从buffer中取样训练
                inputs_hat, inputs, mean, log_var = self.vae_module(inputs)
                lossdict = self.vae_module.loss_function(x=inputs, x_hat=inputs_hat, mean=mean, log_var=log_var, 
                    kld_loss_weight=PymarlAlgorithmConfig.kld_loss_weight, 
                    recons_loss_weight=PymarlAlgorithmConfig.recons_loss_weight, 
                    valid_mask=valid_mask)
                new_input = mean
                if PymarlAlgorithmConfig.DEBUG_DETACH_INPUTS: 
                    new_input = new_input.detach().clone()
                    return torch.zeros_like(new_input), lossdict
                else:
                    return new_input, lossdict
            else:
                # 采样过程
                mean, log_var = self.vae_module.encode(inputs)
                new_input = mean
                if PymarlAlgorithmConfig.DEBUG_DETACH_INPUTS: 
                    return torch.zeros_like(new_input)
                else:
                    return new_input
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape


class my_NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(my_NMAC, self).__init__(scheme, groups, args)

    @Args2tensor_Return2numpy
    def select_actions(self, frame_data, t_env, bs=slice(None), test_mode=False, pr_flag=None):
        # Only select actions for the selected batch elements in bs
        avail_actions = frame_data["avail_actions"]
        qvals = self.forward_(frame_data, test_mode=test_mode)
        pr_flag_ = pr_flag[bs] if pr_flag is not None else None
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode, pr_flag=pr_flag_)
        return chosen_actions

    def forward_(self, frame_data, test_mode=False):
        agent_inputs = self._build_inputs(frame_data, frame_data["time"])
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # (8 env, 6 agent, 30 core dim)
        # self.hidden_states 8, 6, 64
        return agent_outs

    @Args2tensor    #_Return2numpy
    def forward(self, frame_data, valid_mask=None):
        if self.use_vae:
            agent_inputs, vae_lossdict = self._build_inputs(frame_data, frame_data["time"], get_vae_loss=True, valid_mask=valid_mask)
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            return agent_outs, vae_lossdict
        else:
            agent_inputs = self._build_inputs(frame_data, frame_data["time"])
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            return agent_outs

'''
    agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
    summary(model=self, input_size=[(27,348), (27,64)], batch_size=666)
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                  [666, 64]          22,336
            GRUCell-2                  [666, 64]               0
                Linear-3                  [666, 36]           2,340
    ================================================================
    Total params: 24,676
    Trainable params: 24,676
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 41249.72
    Forward/backward pass size (MB): 0.83
    Params size (MB): 0.09
    Estimated Total Size (MB): 41250.65
    ----------------------------------------------------------------
'''