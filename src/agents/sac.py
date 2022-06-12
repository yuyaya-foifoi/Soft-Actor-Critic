import copy
import itertools
import os
import random
import shutil

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset.replay_buffer import ReplayBuffer
from src.dataset.rl_dataset import RLDataset
from src.models.dqn import DQN
from src.models.gradient_policy import GradientPolicy
from src.rl_utils.loss import polyak_average
from src.utils.env import create_env
from src.utils.load_config import CONFIG_PATH, get_config
from src.utils.path import define_log_dir

log_dir = define_log_dir()
config = get_config()

class SAC(LightningModule):
    def __init__(self):

        super().__init__()
        
        torch.manual_seed(config['Base']['Seed'])
        self.device_name = self._set_device()
    
        self.train_env = create_env(is_train=True)
        self.test_env = create_env(is_train=False)

        self._initialize_model()

        if config['Transfer']['is_Transfer']:
            self._transfer()

        self._save_cfg()
        self.buffer = ReplayBuffer(capacity=config['Train']['replay_size'])

        while len(self.buffer) < (config['Train']['update_after']):
            self.play_episode(env=self.train_env, is_initial_buffering=True)

        self.log = {"Qvalue-loss": [], "Policy-loss": [], "reward": []}
        self.reward = 0.0
        self.q_loss = []
        self.policy_loss = []

    def _set_device(self):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        return device

    def _initialize_model(self):

        obs_size = self.train_env.observation_space.shape[0]
        action_dims = self.train_env.action_space.shape[0]
        max_action = self.train_env.action_space.high

        self.q_net1 = DQN(
            config['Model']['hidden_size'],
            obs_size,
            action_dims,
            self.device_name,
        )
        self.q_net2 = DQN(
            config['Model']['hidden_size'],
            obs_size,
            action_dims,
            self.device_name,
        )
        self.policy = GradientPolicy(
            config['Model']['hidden_size'],
            obs_size,
            action_dims,
            max_action,
            self.device_name,
        )

        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)

    def _transfer(self):
        weights = torch.load(config['Transfer']['Weight_path'])

        q_net1_w = weights['q_net1']
        q_net2_w = weights['q_net2']
        policy_w = weights['policy']

        # validation

        assert self.q_net1.state_dict().keys() == q_net1_w.keys()
        assert self.target_q_net1.state_dict().keys() == q_net1_w.keys()

        assert self.q_net2.state_dict().keys() == q_net2_w.keys()
        assert self.target_q_net2.state_dict().keys() == q_net2_w.keys()

        assert self.policy.state_dict().keys() == policy_w.keys()
        assert self.target_policy.state_dict().keys() == policy_w.keys()

        q_net_overwrite_keys = list(set(q_net1_w.keys()))[1:-1]
        policy_overwrite_keys = list(set(policy_w.keys()))[1:-1]

        # overwrite
        self._overwrite_weight(self.q_net1, q_net1_w, q_net_overwrite_keys)
        self._overwrite_weight(self.target_q_net1, q_net1_w, q_net_overwrite_keys)

        self._overwrite_weight(self.q_net2, q_net2_w, q_net_overwrite_keys)
        self._overwrite_weight(self.target_q_net2, q_net2_w, q_net_overwrite_keys)

        self._overwrite_weight(self.policy, policy_w, policy_overwrite_keys)
        self._overwrite_weight(self.target_policy, policy_w, policy_overwrite_keys)

    def _overwrite_weight(self, model, weight, weight_keys: list):
        for weight_key in weight_keys:
            model.state_dict()[weight_key] = weight[weight_key]

    @torch.no_grad()
    def play_episode(self, env, is_initial_buffering=False, is_train=True):
        obs = env.reset()
        done = False

        while not done:
            if is_initial_buffering:
                action = env.action_space.sample()
            if not is_initial_buffering:
                if is_train:
                    action, _ = self.policy(obs, deterministic=False)
                if not is_train:
                    action, _ = self.policy(obs, deterministic=True)
                action = action.cpu().numpy()

            next_obs, reward, done, info = env.step(action)
            exp = (obs, action, reward, done, next_obs)
            obs = next_obs

            if is_initial_buffering or is_train:
                self.buffer.append(exp)

            if not is_train:
                self.reward += reward


    def configure_optimizers(self):
        q_net_params = itertools.chain(
            self.q_net1.parameters(), self.q_net2.parameters()
        )
        q_net_optimizer = Adam(q_net_params, lr=config['Train']['critic_lr'])
        policy_optimizer = Adam(
            self.policy.parameters(), lr=config['Train']['actor_lr']
        )
        return [q_net_optimizer, policy_optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, config['Train']['samples_per_epoch'])
        dataloader = DataLoader(dataset=dataset, batch_size=1)
        return dataloader

    def training_step(self, batch, batch_idx, optimizer_idx):
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        if optimizer_idx == 0:
            action_values1 = self.q_net1(states, actions)
            action_values2 = self.q_net2(states, actions)

            target_actions, target_log_probs = self.target_policy(next_states)

            next_actions_values = torch.min(
                self.target_q_net1(next_states, target_actions),
                self.target_q_net2(next_states, target_actions),
            )

            next_actions_values[dones] = 0.0

            expected_action_values = rewards + config['Train']['gamma'] * (
                next_actions_values - config['Train']['alpha'] * target_log_probs
            )

            loss_q1 = ((action_values1 - expected_action_values)**2).mean()
            loss_q2 = ((action_values2 - expected_action_values)**2).mean()

            q_loss_total = loss_q1 + loss_q2
            # self.log["Qvalue-loss"].append(q_loss_total)

            self.q_loss.append(q_loss_total.item())

            return q_loss_total

        elif optimizer_idx == 1:

            actions, log_probs = self.policy(states)

            action_values = torch.min(
                self.target_q_net1(next_states, actions),
                self.target_q_net2(next_states, actions),
            )

            policy_loss = (
                config['Train']['alpha'] * log_probs - action_values
            ).mean()
            # self.log["Policy-loss"].append(policy_loss)

            self.policy_loss.append(policy_loss.item())

            return policy_loss

    def training_epoch_end(self, training_step_outputs):

        self.play_episode(env=self.train_env, is_train=True)
        self.play_episode(env=self.test_env, is_train=False)
        self._update_model()
        self._epoch_end_operation()

    def _update_model(self):
        polyak_average(self.q_net1, self.target_q_net1, rho=config['Train']['polyak_rho'])
        polyak_average(self.q_net2, self.target_q_net2, rho=config['Train']['polyak_rho'])
        polyak_average(self.policy, self.target_policy, rho=config['Train']['polyak_rho'])

    def _epoch_end_operation(self):
        self._logging()
        self._clear_log()
        self._dump_log()
        self._dump_weight()

    def _logging(self):
        self.log["Qvalue-loss"].append(np.mean(self.q_loss))
        self.log["Policy-loss"].append(np.mean(self.policy_loss))
        self.log["reward"].append(self.reward)

    def _clear_log(self):
        self.reward = 0.0
        self.q_loss = []
        self.policy_loss = []

    def _save_cfg(self):
        shutil.copy(CONFIG_PATH, log_dir)

    def _save_hparams(self):
        path = os.path.join(log_dir, "hparams.pkl")
        torch.save(dict(self.hparams), path)

    def _dump_log(self):
        path = os.path.join(log_dir, "log.pkl")
        torch.save(self.log, path)

    def _dump_weight(self):
        path = os.path.join(log_dir, "weights.pkl")
        weights = {
            "q_net1": self.q_net1.state_dict(),
            "q_net2": self.q_net2.state_dict(),
            "policy": self.policy.state_dict(),
        }
        torch.save(weights, path)
