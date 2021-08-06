from model import Model
import numpy as np
import torch
from torch import from_numpy
from atari_wrappers import make_atari
from utils import make_state


class Worker:
    def __init__(self,
                 id,
                 state_shape,
                 n_actions,
                 env_name,
                 global_model,
                 shared_optimizer,
                 gamma,
                 ent_coeff,
                 update_period,
                 max_steps_per_episode):

        self.id = id
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.update_period = update_period
        self.env_name = env_name
        self.env = make_atari(self.env_name)
        self.model = Model(self.state_shape, self.n_actions)
        self.global_model = global_model
        self.shared_optimizer = shared_optimizer
        self.mse_loss = torch.nn.MSELoss()
        self.ep = 0
        self.max_steps_per_episode = max_steps_per_episode

    def get_actions_and_values(self, state):

        state = np.expand_dims(state, 0)
        state = from_numpy(state).byte()
        with torch.no_grad():
            dist, value = self.model(state)
            action = dist.sample()
        return action.numpy(), value.numpy().squeeze()

    def sync_thread_spec_params(self):
        self.model.load_state_dict(self.global_model.state_dict())

    @staticmethod
    def share_grads_to_global_models(local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

    def step(self):
        print(f"Worker: {self.id} started.")
        running_reward = 0
        state = np.zeros(self.state_shape, dtype=np.uint8)
        obs = self.env.reset()
        state = make_state(state, obs, True)
        next_state = None
        episode_reward = 0
        while True:
            self.shared_optimizer.zero_grad()  # Reset global gradients

            self.sync_thread_spec_params()  # Synchronize thread-specific parameters

            states, actions, rewards, dones = [], [], [], []

            for step in range(1, 1 + self.max_steps_per_episode):
                action, _ = self.get_actions_and_values(state)
                next_obs, reward, done, _ = self.env.step(action)
                # self.env.render()
                states.append(state)
                actions.append(action)
                rewards.append(np.sign(reward))
                dones.append(done)

                episode_reward += reward
                next_state = make_state(state, next_obs, False)
                state = next_state

                if done:
                    obs = self.env.reset()
                    state = make_state(state, obs, True)

                    self.ep += 1
                    if self.ep == 1:
                        running_reward = episode_reward
                    else:
                        running_reward = 0.9 * running_reward + 0.1 * episode_reward

                    print(f"\nW{self.id} Ep {self.ep}: {running_reward:.0f}")
                    episode_reward = 0

                if step % self.update_period == 0 or done:
                    break

            _, R = self.get_actions_and_values(next_state)
            returns = []
            for r, d in zip(rewards[::-1], dones[::-1]):
                R = r + self.gamma * R * (1 - d)
                returns.insert(0, R)

            states = torch.Tensor(states).view(-1, *self.state_shape)
            actions = torch.Tensor(actions)
            returns = torch.Tensor(returns).view(-1, 1)

            dist, values = self.model(states)
            log_probs = dist.log_prob(actions.squeeze(1))
            advs = returns - values

            pg_loss = -(log_probs * advs.squeeze(1).detach()).mean()
            value_loss = self.mse_loss(values, returns)

            total_loss = pg_loss - self.ent_coeff * dist.entropy().mean() + 0.5 * value_loss

            total_loss.backward()

            self.share_grads_to_global_models(self.model, self.global_model)

            self.shared_optimizer.step()
