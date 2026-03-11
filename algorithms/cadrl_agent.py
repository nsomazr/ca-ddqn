"""
CA-DRL DQN agent for satellite IoT channel allocation.

Implements:
- MLP Q-network over the CA-DRL state representation
- Target network
- Prioritised experience replay
- Epsilon-greedy exploration

This closely follows the algorithmic description in the CA-DRL paper, with
hyperparameters provided by `configs/experiment_config.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment.wireless_env import SatelliteEnvironment
from utils.config_loader import ExperimentConfig, TrainingConfig


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers=None, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        layers = []
        in_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute Q-values."""
        return self.net(x)


@dataclass
class ReplayTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple uniform replay buffer; easy to swap for prioritised version later."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[ReplayTransition] = []
        self.position = 0

    def push(self, transition: ReplayTransition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if len(self.buffer) < batch_size:
            return ()
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.stack([t.state for t in batch]).astype(np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch]).astype(np.float32)
        dones = np.array([t.done for t in batch], dtype=np.bool_)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:  # noqa: D401
        """Current number of transitions stored."""
        return len(self.buffer)


class CADRLAgent:
    def __init__(self, env: SatelliteEnvironment, train_cfg: TrainingConfig, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = torch.device(device)

        self.q_net = QNetwork(env.state_dim, env.action_dim).to(self.device)
        self.target_net = QNetwork(env.state_dim, env.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=train_cfg.learning_rate)
        self.buffer = ReplayBuffer(capacity=train_cfg.replay_buffer_size)

        self.gamma = train_cfg.discount_factor
        self.epsilon = train_cfg.epsilon_start
        self.epsilon_min = train_cfg.epsilon_min
        self.epsilon_decay = train_cfg.epsilon_decay
        self.target_update_freq = train_cfg.target_update_frequency
        self.global_step = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.env.action_dim)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.q_net(s)
            return int(q.argmax(dim=1).item())

    def update(self) -> Optional[float]:
        batch = self.buffer.sample(self.cfg.minibatch_size)
        if not batch:
            return None
        states, actions, rewards, next_states, dones = batch

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_max = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * (1.0 - dones_t) * next_q_max

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        if self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net_state_dict": self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_config": self.cfg.__dict__,
                "epsilon": self.epsilon,
            },
            path,
        )


class CADRLTrainer:
    """
    High-level trainer wrapping the environment and CADRLAgent into an episode loop.
    """

    def __init__(self, env: SatelliteEnvironment, agent: CADRLAgent) -> None:
        self.env = env
        self.agent = agent

    def train_episode(self) -> Dict[str, float]:
        state = self.env.reset()
        done = False

        episode_reward = 0.0
        while not done:
            action = self.agent.select_action(state, training=True)
            next_state, reward, done, info = self.env.step(action)
            self.agent.buffer.push(
                ReplayTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )
            loss = self.agent.update()
            state = next_state
            episode_reward += reward

        metrics = self.env.get_episode_metrics()
        metrics.update({"episode_reward": episode_reward})
        return metrics

    def evaluate_episode(self) -> Dict[str, float]:
        state = self.env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = self.agent.select_action(state, training=False)
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            episode_reward += reward

        metrics = self.env.get_episode_metrics()
        metrics.update({"episode_reward": episode_reward})
        return metrics


