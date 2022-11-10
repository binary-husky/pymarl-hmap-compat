import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.one_step_matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch
from torch.optim import RMSprop, Adam
import numpy as np
from torch.distributions import Categorical
from utils.th_utils import get_parameters_num
from UTIL.tensor_ops import gather_righthand, __hash__, _2tensor, __hashn__
# from UTIL.tensor_ops import dump_sychronize_data, sychronize_experiment, sychronize_internal_hashdict
from commom.norm import DynamicNormFix
from config import GlobalConfig
from controllers.my_n_controller import PymarlAlgorithmConfig
def get_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    else: return x

class NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = torch.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            if mac.use_vae:
                mac_rnn_param, mac_vae_param = mac.parameters()
                mixer_param = self.mixer.parameters()
                # self.optimiser = Adam(params={'mac_rnn_param':mac_rnn_param, 'mixer_param':mixer_param, 'mac_vae_param':mac_vae_param},  lr=args.lr)
                self.optimiser = Adam(params=[
                    {'params':mac_rnn_param, 'lr':args.lr}, 
                    {'params':mixer_param, 'lr':args.lr}, 
                    {'params':mac_vae_param, 'lr':args.lr*PymarlAlgorithmConfig.vae_lr_ratio} ])
                self.params = []
                for g in self.optimiser.param_groups:
                    self.params.extend(g['params'])

            else:
                self.optimiser = Adam(params=self.params,  lr=args.lr)
        else:
            assert False
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)    # shit-like code

        if self.mac.use_normalization:
            # 有两个mac需要共享归一模块，本来可以在self.mac内部初始化，但是注意上面有个艹蛋的copy.deepcopy
            assert self.target_mac.use_normalization
            assert self.mac.input_shape == self.target_mac.input_shape
            self._batch_norm = DynamicNormFix(self.mac.input_shape, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
            self.mac._batch_norm = self._batch_norm
            self.target_mac._batch_norm = self._batch_norm
            self.state_batch_norm = DynamicNormFix(args.state_shape, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0

        # torch.autograd.set_detect_anomaly(True)
        
    def train(self, traj_manager, current_pool_subset, t_env: int, episode_num: int):
        # Get the relevant quantities
        # rewards = traj_manager.parse(traj_pool=current_pool_subset,       key_name="rewards",       method="remove tail")

        # I use stack_padding here, using ***np.nan*** to pad!
        # with torch.no_grad():
        rewards =           _2tensor(traj_manager.parse(traj_pool=current_pool_subset, key_name="rewards",           method="normal",     padding=np.nan))
        rewards_tailed_nan= _2tensor(traj_manager.parse(traj_pool=current_pool_subset, key_name="rewards",           method="+nan",       padding=np.nan))
        actions =           _2tensor(traj_manager.parse(traj_pool=current_pool_subset, key_name="actions",           method="+zero",      padding=0))
        obs =               _2tensor(traj_manager.parse(traj_pool=current_pool_subset, key_name="obs",               method="+$tobs",     padding=0))
        state =             _2tensor(traj_manager.parse(traj_pool=current_pool_subset, key_name="state",             method="+$tstate",   padding=0))
        avail_actions_m1 =  _2tensor(traj_manager.parse(traj_pool=current_pool_subset, key_name="actions_onehot -1", method="+dupl",      padding=0))
        avail_actions =     _2tensor(traj_manager.parse(traj_pool=current_pool_subset, key_name="avail_actions",     method="+dupl",      padding=0))

        if self.mac.use_normalization:
            state = self.state_batch_norm(state)

        valid_mask = ~torch.isnan(rewards) # torch.cat((mask,mask[:, 0:1]*0), axis=1)
        valid_mask_tailed = ~torch.isnan(rewards_tailed_nan) # torch.cat((mask,mask[:, 0:1]*0), axis=1)

        batch_size = len(current_pool_subset)
        # max_seq_length = 
        # Calculate estimated Q-Values
        mac_out = []
        vae_loss = []
        vae_loss_target = []
        # 网络1：主网络
        self.mac.init_hidden(batch_size)
        # valid_mask ($n_th_traj, $time)
        # with torch.no_grad():
        for t in range(obs.shape[1]):   # RNN should run repeating over time
            agent_outs = self.mac.forward({ "obs": obs[:, t], "actions_onehot -1": avail_actions_m1[:, t].clone(), "time": t}, valid_mask=valid_mask_tailed[:, t])

            if self.mac.use_vae:
                agent_outs, vae_loss_dict = agent_outs
                vae_loss.append(vae_loss_dict)

            mac_out.append(agent_outs)  # agent_outs. Q = ( $n_thread, $n_agent, $n_actions)

        mac_out = torch.stack(mac_out, axis=1)  # Concat over time, mac_out Q = [$n_thread(n_batch)=128, $time=60, $n_agent, $n_actions]
        
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = gather_righthand(mac_out, index=actions.unsqueeze(-1)).squeeze(-1) 

        # Calculate the Q-Values necessary for the target
        with torch.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch_size)
            # 网络2：target网络
            for t in range(obs.shape[1]):
                target_agent_outs = self.target_mac.forward({"obs": obs[:, t], "actions_onehot -1": avail_actions_m1[:, t].clone(), "time": t}, valid_mask=valid_mask_tailed[:, t])
                
                if self.mac.use_vae:
                    target_agent_outs, vae_loss_dict = target_agent_outs
                    vae_loss_target.append(vae_loss_dict)

                target_mac_out.append(target_agent_outs)
            target_mac_out = torch.stack(target_mac_out, axis=1)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -float('inf') #np.inf
            cur_max_actions = torch.argmax(mac_out_detach, axis=-1)        # mac_out_detach ($n_thread(n_batch)=128, $time = 60, $n_agent = 3, $n_actions = 9)
            # target_max_qvals = gather_righthand(target_mac_out, index=np.expand_dims(cur_max_actions,-1)).squeeze(-1)
            target_max_qvals = gather_righthand(target_mac_out, index=cur_max_actions.unsqueeze(-1)).squeeze(-1)
            # torch.Size([128, 60, 3, 9]),  torch.Size([128, 60, 3, 1])
            # Calculate n-step Q-Learning targets 
            target_max_qvals = self.target_mixer(target_max_qvals, state)

            if getattr(self.args, 'q_lambda', False):
                assert False
                qvals = torch.gather(target_mac_out, 3, actions).squeeze(3)
                qvals = self.target_mixer(qvals, state)
                targets = build_q_lambda_targets(rewards, terminated, None, target_max_qvals, qvals, self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, None, None, target_max_qvals, self.args.n_agents, self.args.gamma, self.args.td_lambda)

        chosen_action_qvals = chosen_action_qvals[:,:-1,...] # Remove the last dim !!!!
        state = state[:,:-1,...] # Remove the last dim !!!!

        if self.mac.use_vae:
            # valid_mask ($n_th_traj, $time)
            loss_vae = torch.stack([x['loss'] for x in vae_loss_target]+[x['loss'] for x in vae_loss]).mean()
            Reconstruction_Loss = torch.stack([x['Reconstruction_Loss'] for x in vae_loss_target]+[x['Reconstruction_Loss'] for x in vae_loss]).mean()
            kld_Loss = torch.stack([x['KLD'] for x in vae_loss_target]+[x['KLD'] for x in vae_loss]).mean()
        else:
            loss_vae = 0
            Reconstruction_Loss = 0
            kld_Loss = 0
        # Mixer
        mixed_qvals = self.mixer(chosen_action_qvals, state)
        # mixed_qvals[~valid_mask] = 0 # Remove invalid data content

        td_error = (mixed_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)

        loss = td_error[valid_mask].mean() + loss_vae
        # sychronize_experiment('syc_loss', loss)
        # print('syc_loss successful train')

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        self.mac.free_hidden_state()
        self.target_mac.free_hidden_state()

        if GlobalConfig.activate_logger: 
            GlobalConfig.mcv.rec(get_item(loss_vae), 'LossVae') 
            GlobalConfig.mcv.rec(get_item(Reconstruction_Loss), 'ReconLoss') 
            GlobalConfig.mcv.rec(get_item(kld_Loss), 'KldLoss') 
            GlobalConfig.mcv.rec(get_item(loss), 'AllLoss') 

        # dump_sychronize_data()
        

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        # self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.mac.use_normalization:
            self._batch_norm.cuda()
            self.state_batch_norm.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
        if self.mac.use_normalization:
            torch.save(self._batch_norm.state_dict(), "{}/bn.th".format(path))
            torch.save(self.state_batch_norm.state_dict(), "{}/sbn.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(torch.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        if self.mac.use_normalization:
            self._batch_norm.load_state_dict(torch.load("{}/bn.th".format(path), map_location=lambda storage, loc: storage))
            self.state_batch_norm.load_state_dict(torch.load("{}/sbn.th".format(path), map_location=lambda storage, loc: storage))
