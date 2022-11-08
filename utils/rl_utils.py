import torch
import numpy as np
from config import GlobalConfig
# from UTIL.tensor_ops import dump_sychronize_data, sychronize_experiment, sychronize_internal_hashdict

'''
tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2489, 0.0043, 0.0043,
        0.0043, 0.0043, 0.0043, 0.2444, 0.0064, 0.1178, 0.0085, 0.4880, 0.0128,
        0.4837, 0.0171, 0.1079, 0.2329, 0.0192, 0.0192, 0.0192, 0.0192, 0.2292,
        0.4773, 0.0235, 0.0235, 0.0235, 0.0235, 0.0235, 0.2244, 0.0256, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000], device='cuda:0')
'''

def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1.  第一维B是episode的数量，第二维是时间，第三位是核心维
    if target_qs.dim()>=3 and target_qs.shape[-1]==1:
        target_qs = target_qs.squeeze(-1)

    res2 = build_td_lambda_targets_human_read_efficient(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda, zero_q_terminal=False)
    res2 = res2.unsqueeze(-1) # 去除最后一步
    res2 = res2[:, :-1]

    return res2

# def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
#     # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
#     # Initialise  last  lambda -return  for  not  terminated  episodes
#     ret = target_qs.new_zeros(*target_qs.shape)
#     ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
#     # Backwards  recursive  update  of the "forward  view"
#     for t in range(ret.shape[1] - 2, -1,  -1):
#         ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
#                     * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
#     # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
#     return ret[:, 0:-1]

'''
rewards.S torch.Size([128, 70, 1])
'''

def build_gae_targets(rewards, masks, values, gamma, lambd):
    B, T, _ = values.size()
    T-=1
    advantages = torch.zeros(B, T, 1).to(device=values.device)
    advantage_t = torch.zeros(B, 1).to(device=values.device)

    for t in reversed(range(T)):
        delta = rewards[:, t] + values[:, t+1] * gamma * masks[:, t] - values[:, t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[:, t]
        advantages[:, t] = advantage_t

    returns = values[:, :T] + advantages
    return advantages, returns


def build_q_lambda_targets(rewards, terminated, mask, exp_qvals, qvals, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = exp_qvals.new_zeros(*exp_qvals.shape)
    ret[:, -1] = exp_qvals[:, -1] * (1 - torch.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        reward = rewards[:, t] + exp_qvals[:, t] - qvals[:, t] #off-policy correction
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (reward + (1 - td_lambda) * gamma * exp_qvals[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def build_target_q(td_q, target_q, mac, mask, gamma, td_lambda, n):
    aug = torch.zeros_like(td_q[:, :1])

    #Tree diagram
    mac = mac[:, :-1]
    tree_q_vals = torch.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for _ in range(n):
        tree_q_vals += t1 * coeff
        t1 = torch.cat(((t1 * mac)[:, 1:], aug), dim=1)
        coeff *= gamma * td_lambda
    return target_q + tree_q_vals
        
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

def add_tail_last_dim(tensor, padding=np.nan):
    tail = tensor[..., -1:]+padding
    return torch.cat((tensor,tail),-1)

def build_td_lambda_targets_human_read_efficient(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda, zero_q_terminal=False):
    if rewards.shape[-1] == 1: 
        # reward 不带nan padding，手动添加
        assert GlobalConfig.runner != 'efficient_parallel_runner'
        rewards = torch.where(mask.bool(), input=rewards, other=rewards+float('nan'))
        rewards = rewards.squeeze(-1)
    else:
        # reward 自带nan padding
        assert GlobalConfig.runner == 'efficient_parallel_runner'

    rewards = add_tail_last_dim(rewards)
    # 基线，所有量的标准是s_{t}的时间
    ret = torch.zeros_like(rewards) #.new_zeros(*target_qs.shape)
    t_max = ret.shape[1]
    lambda_gamma_ = (td_lambda * gamma)
    gamma_minus_lambda_gamma_ = (1 - td_lambda) * gamma

    valid_mask_ = ~torch.isnan(rewards) # torch.cat((mask,mask[:, 0:1]*0), axis=1)
    # terminated_ = torch.cat((terminated,terminated[:, 0:1]*0), axis=1)

    # warning!
    # Q(s_{Terminal}-1) = r_{Terminal}
    target_qs_ = target_qs.clone()
    target_qs_[~valid_mask_] = 0    # nan -> 0, so that mask can handle (nan*0=nan)
    rewards[~valid_mask_] = 0    # nan -> 0, so that mask can handle (nan*0=nan)

    # warning!
    # Q(s_{Terminal})   = 0  if zero_q_terminal

    # note! the reward moment should be the exect reward obtained during s_{t} -> s_{t+1}
    # In papers, we refer to this reward as R_{t}, but in program, it's rewards[t]
    for t in reversed(range(t_max)):
        # t = [t_max-1, t_max-2, ..., 0]
        if t==t_max-1: 
            t_terminated_ = valid_mask_[:,t]
        else:
            t_terminated_ = valid_mask_[:,t] & ~valid_mask_[:,t+1]
        if zero_q_terminal: 
            target_qs_[t_terminated_, t] = 0

        if t==t_max-1: 
            continue    # note here!

        part1_filter = valid_mask_[:,t]
        # part2_filter = (mask_[:,t] != 0) & (mask_[:,t+1] != 0)
        part2_filter = part1_filter & valid_mask_[:,t+1]

        ret[:, t] = (
            rewards[:,t] 
            + lambda_gamma_ * ret[:, t+1] * part1_filter
            + gamma_minus_lambda_gamma_ * target_qs_[:, t+1] * part2_filter
        )

    return ret

def build_td_lambda_targets_human_read(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda, zero_q_terminal=False):
    # 基线，所有量的标准是s_{t}的时间

    ret = torch.zeros_like(target_qs) #.new_zeros(*target_qs.shape)
    b_max = ret.shape[0]
    t_max = ret.shape[1]

    lambda_gamma_ = (td_lambda * gamma)
    gamma_minus_lambda_gamma_ = (1 - td_lambda) * gamma

    mask_ = torch.cat((mask,mask[:, 0:1]*0), axis=1)
    terminated_ = torch.cat((terminated,terminated[:, 0:1]*0), axis=1)

    # Q(s_{Terminal}-1) = r_{Terminal}
    # 清除无意义的Qs值（置NaN）
    target_qs_ = target_qs.clone()
    target_qs_[mask_==0] = np.nan

    # warning!
    # Q(s_{Terminal})   = 0  if zero_q_terminal
    if zero_q_terminal:
        target_qs_[terminated_==1] = 0

    for t in reversed(range(t_max)):
        for b in range(b_max):
            # note: the point is, the reward moment should be the exect reward obtained during s_{t} -> s_{t+1}
            # In papers, we refer to this reward as R_{t}, but in program, it's rewards[t]
            if t==t_max-1 or mask_[b,t] == 0: 
                continue    # default 0
            elif mask_[b,t+1] == 0:
                ret[b, t] = rewards[b,t] + lambda_gamma_ * ret[b, t+1]
            else:
                ret[b, t] = rewards[b,t] + lambda_gamma_ * ret[b, t+1] + gamma_minus_lambda_gamma_ * target_qs_[b, t+1]
    # return ret[:, 0:-1]
    return ret
