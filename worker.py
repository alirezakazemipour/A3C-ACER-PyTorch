from model import Actor, Critic
import gym
import numpy as np
import torch
from torch import from_numpy
import sys

class Worker:
    def __init__(self,
                 id,
                 n_states,
                 n_actions,
                 action_bounds,
                 env_name,
                 n_hiddens,
                 global_actor,
                 global_critic,
                 shared_actor_optimizer,
                 shared_critic_optimizer,
                 gamma,
                 ent_coeff):
        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_bounds = action_bounds
        self.n_hiddens = n_hiddens
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        self.local_actor = Actor(self.n_states, self.n_actions, self.action_bounds, self.n_hiddens)
        self.local_critic = Critic(self.n_states, self.n_hiddens)

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.shared_actor_optimizer = shared_actor_optimizer
        self.shared_critic_optimizer = shared_critic_optimizer

        self.mse_loss = torch.nn.MSELoss()

    def get_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            dist = self.local_actor(state)
            action = dist.sample()
        action = np.clip(action.numpy(), self.action_bounds[0], self.action_bounds[1])
        # print(action)
        return action

    def get_value(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            value = self.local_critic(state)
        return value.numpy()

    def sync_thread_spec_params(self):
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    @staticmethod
    def share_grads_to_global_models(local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                # print(global_param.grad)
                return
            # print("here2")
            global_param._grad = local_param.grad

    def step(self):
        print(f"Worker: {self.id} started.")
        running_reward = 0
        while True:
            self.sync_thread_spec_params()  # Synchronize thread-specific parameters

            states, actions, rewards, dones, next_states = [], [], [], [], []
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.get_action(state)
                # print(action[0])
                next_state, reward, done, _ = self.env.step(action[0])
                # self.env.render()
                states.append(state)
                actions.append(action)
                rewards.append(reward / 8 + 1)
                dones.append(dones)
                next_states.append(next_state)
                episode_reward += reward

            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            states = torch.Tensor(states).view(-1, self.n_states)
            actions = torch.Tensor(actions).view(-1, 1)
            returns = torch.Tensor(returns).view(-1, 1)

            dist = self.local_actor(states)
            log_probs = dist.log_prob(actions)

            values = self.local_critic(states)
            advs = returns - values

            pg_loss = -(log_probs * advs.detach()).mean()
            value_loss = self.mse_loss(values, returns)

            actor_loss = pg_loss - self.ent_coeff * dist.entropy().mean()

            self.shared_actor_optimizer.zero_grad()  # Reset global gradients
            self.shared_critic_optimizer.zero_grad()

            actor_loss.backward()
            value_loss.backward()

            self.share_grads_to_global_models(self.local_actor, self.global_actor)
            self.share_grads_to_global_models(self.local_critic, self.global_critic)

            self.shared_actor_optimizer.step()
            self.shared_critic_optimizer.step()

            running_reward = 0.9 * running_reward + 0.1 * episode_reward
            print(f"Worker {self.id}: {running_reward:.0f}")
            # print(loss)
