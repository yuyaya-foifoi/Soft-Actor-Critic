import copy
import itertools
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.dataset.replay_buffer import ReplayBuffer
from src.dataset.rl_dataset import RLDataset
from src.models.dqn import DQN
from src.models.gradient_policy import GradientPolicy
from src.rl_utils.averaging import polyak_average
from src.utils.path import define_log_dir
from src.utils.video import create_environment

log_dir = define_log_dir()


class SAC(LightningModule):
    def __init__(
        self,
        device,
        capacity=100_000,
        batch_size=256,
        lr=1e-3,
        hidden_size=256,
        gamma=0.99,
        loss_fn=F.smooth_l1_loss,
        optim=AdamW,
        samples_per_epoch=1_000,
        tau=0.05,
        epsilon=0.05,
        alpha=0.02,
        transfer=False,
        transfer_path=None,
    ):

        super().__init__()

        self.save_hyperparameters()
        self.env = create_environment()
        self._initialize_model()

        self._save_cfg()
        self._save_hparams()
        self.buffer = ReplayBuffer(capacity=capacity)

        while len(self.buffer) < self.hparams.samples_per_epoch:
            self.play_episode(is_logging_reward=False)

        self.log = {"Qvalue-loss": [], "Policy-loss": [], "reward": []}
        self.reward = 0.0
        self.q_loss = 0.0
        self.policy_loss = 0.0

    def _initialize_model(self):

        obs_size = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]
        max_action = self.env.action_space.high

        self.q_net1 = DQN(
            self.hparams.hidden_size,
            obs_size,
            action_dims,
            self.hparams.device,
        )
        self.q_net2 = DQN(
            self.hparams.hidden_size,
            obs_size,
            action_dims,
            self.hparams.device,
        )
        self.policy = GradientPolicy(
            self.hparams.hidden_size,
            obs_size,
            action_dims,
            max_action,
            self.hparams.device,
        )

        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)
        self.target_policy = copy.deepcopy(self.policy)

    @torch.no_grad()
    def play_episode(self, policy=None, is_logging_reward=False):
        obs = self.env.reset()
        done = False

        while not done:
            if policy and random.random() > self.hparams.epsilon:
                action, _ = self.policy(obs)
                action = action.cpu().numpy()
            else:
                action = self.env.action_space.sample()

            next_obs, reward, done, info = self.env.step(action)
            exp = (obs, action, reward, done, next_obs)
            self.buffer.append(exp)
            obs = next_obs

            if is_logging_reward:
                self.reward += reward

    def forward(self, x):
        """
        foward 関数は主に評価時に用いる
        評価時には決定論的なpolicy関数から行動を出力し,
        エージェントがその行動を実行する
        """
        output = self.policy(x)
        return output

    def configure_optimizers(self):
        q_net_params = itertools.chain(
            self.q_net1.parameters(), self.q_net2.parameters()
        )
        q_net_optimizer = self.hparams.optim(q_net_params, lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(
            self.policy.parameters(), lr=self.hparams.lr
        )
        return [q_net_optimizer, policy_optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
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

            expected_action_values = rewards + self.hparams.gamma * (
                next_actions_values - self.hparams.alpha * target_log_probs
            )

            q_loss1 = self.hparams.loss_fn(
                action_values1, expected_action_values
            )
            q_loss2 = self.hparams.loss_fn(
                action_values2, expected_action_values
            )

            q_loss_total = q_loss1 + q_loss2
            # self.log["Qvalue-loss"].append(q_loss_total)

            self.q_loss += q_loss_total.item()

            return q_loss_total

        elif optimizer_idx == 1:

            actions, log_probs = self.policy(states)

            action_values = torch.min(
                self.target_q_net1(next_states, actions),
                self.target_q_net2(next_states, actions),
            )

            policy_loss = (
                self.hparams.alpha * log_probs - action_values
            ).mean()
            # self.log["Policy-loss"].append(policy_loss)

            self.policy_loss += policy_loss.item()

            return policy_loss

    def training_epoch_end(self, training_step_outputs):

        self.play_episode(policy=self.policy, is_logging_reward=True)
        self._update_model()
        self._epoch_end_operation()

    def _update_model(self):
        polyak_average(self.q_net1, self.target_q_net1, tau=self.hparams.tau)
        polyak_average(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_average(self.policy, self.target_policy, tau=self.hparams.tau)

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
        self.q_loss = 0.0
        self.policy_loss = 0.0

    def _save_cfg(self):
        shutil.copy("./config/config.yml", log_dir)

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
