from NN.model import Actor, Critic
import gym
import numpy as np
import torch
from torch import from_numpy


class Worker(torch.multiprocessing.Process):
    def __init__(self,
                 id,
                 n_states,
                 n_actions,
                 action_bounds,
                 env_name,
                 n_hiddens,
                 global_actor,
                 global_critic,
                 shared_actor_opt,
                 shared_critic_opt,
                 gamma,
                 ent_coeff,
                 max_grad):
        super(Worker, self).__init__()
        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_bounds = action_bounds
        self.n_hiddens = n_hiddens
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.max_grad = max_grad
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        self.local_actor_model = Actor(self.n_states, self.n_actions, self.action_bounds, self.n_hiddens)
        self.local_critic_model = Critic(self.n_states)

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.shared_actor_opt = shared_actor_opt
        self.shared_critic_opt = shared_critic_opt

        self.mse_loss = torch.nn.MSELoss()
        self.ep = 0

    def get_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            dist = self.local_actor_model(state)
            action = dist.sample()
        action = np.clip(action.numpy(), self.action_bounds[0], self.action_bounds[1])
        return action

    def get_value(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            value = self.local_critic_model(state)
        return value.numpy()

    @staticmethod
    def sync_thread_spec_params(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    @staticmethod
    def share_grads_to_global_models(local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

    def run(self):
        print(f"Worker: {self.id} started.")
        running_reward = 0
        state = self.env.reset()
        next_state = None
        episode_reward = 0
        while True:
            self.shared_actor_opt.zero_grad()
            self.shared_critic_opt.zero_grad()
            # Synchronize thread-specific parameters
            self.sync_thread_spec_params(self.global_actor, self.local_actor_model)
            self.sync_thread_spec_params(self.global_critic, self.local_critic_model)

            states, actions, rewards, dones = [], [], [], []

            for step in range(1, 1 + self.env.spec.max_episode_steps):
                action = self.get_action(state)
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

                    break

            R = self.get_value(next_state)[0]
            returns = []
            for r, d in zip(rewards[::-1], dones[::-1]):
                R = r + self.gamma * R * (1 - d)
                returns.insert(0, R)

            states = torch.Tensor(states).view(-1, self.n_states)
            actions = torch.Tensor(actions).view(-1, self.n_actions)
            returns = torch.Tensor(returns)

            dist = self.local_actor_model(states)
            log_probs = dist.log_prob(actions)

            values = self.local_critic_model(states)
            advs = returns - values

            pg_loss = -(log_probs * advs.detach()).mean()
            value_loss = self.mse_loss(returns, values)

            actor_loss = pg_loss - self.ent_coeff * dist.entropy().mean()

            actor_loss.backward()
            value_loss.backward()

            self.share_grads_to_global_models(self.local_actor_model, self.global_actor)
            self.share_grads_to_global_models(self.local_critic_model, self.global_critic)

            torch.nn.utils.clip_grad_norm_(self.local_actor_model.parameters(), self.max_grad)
            torch.nn.utils.clip_grad_norm_(self.local_critic_model.parameters(), self.max_grad)

            self.shared_actor_opt.step()
            self.shared_critic_opt.step()
