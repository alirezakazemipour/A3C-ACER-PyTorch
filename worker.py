from model import Actor, Critic
import gym
import numpy as np
import torch
from torch import from_numpy


class Worker:
    def __init__(self,
                 id,
                 n_states,
                 n_actions,
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
        self.n_hiddens = n_hiddens
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        self.local_actor = Actor(self.n_states, self.n_actions, self.n_hiddens * 2)
        self.local_critic = Critic(self.n_states)

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.shared_actor_optimizer = shared_actor_optimizer
        self.shared_critic_optimizer = shared_critic_optimizer

        self.mse_loss = torch.nn.MSELoss()
        self.ep = 0

    def get_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            dist = self.local_actor(state)
            action = dist.sample()
        return action.numpy()

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
                return
            global_param._grad = local_param.grad

    def step(self):
        print(f"Worker: {self.id} started.")
        running_reward = 0
        state = self.env.reset()
        next_state = None
        episode_reward = 0
        while True:
            self.shared_actor_optimizer.zero_grad()  # Reset global gradients
            self.shared_critic_optimizer.zero_grad()

            self.sync_thread_spec_params()  # Synchronize thread-specific parameters

            states, actions, rewards, dones = [], [], [], []

            for step in range(1, 1 + self.env.spec.max_episode_steps):
                action = self.get_action(state)
                # print(action)
                next_state, reward, done, _ = self.env.step(action[0])

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                episode_reward += reward
                state = next_state

                if done:
                    state = self.env.reset()
                    self.ep += 1
                    if self.ep == 1:
                        running_reward = episode_reward
                    else:
                        running_reward = 0.99 * running_reward + 0.01 * episode_reward
                    if self.id == 0:
                        print(f"\nWorker {self.id}: {running_reward:.0f}")
                    episode_reward = 0

                if step % 10 == 0 or done:
                    break

            R = self.get_value(next_state)[0]
            returns = []
            for r, d in zip(rewards[::-1], dones[::-1]):
                R = r + self.gamma * R * (1 - d)
                returns.insert(0, R)

            states = torch.Tensor(states).view(-1, self.n_states)
            actions = torch.Tensor(actions).view(-1, 1)
            returns = torch.Tensor(returns).view(-1, 1)

            dist = self.local_actor(states)
            log_probs = dist.log_prob(actions.squeeze(1))

            values = self.local_critic(states)
            advs = returns - values

            pg_loss = -(log_probs * advs.detach()).mean()
            value_loss = self.mse_loss(values, returns)

            actor_loss = pg_loss - self.ent_coeff * dist.entropy().mean()

            actor_loss.backward()
            value_loss.backward()

            self.share_grads_to_global_models(self.local_actor, self.global_actor)
            self.share_grads_to_global_models(self.local_critic, self.global_critic)

            self.shared_actor_optimizer.step()
            self.shared_critic_optimizer.step()
