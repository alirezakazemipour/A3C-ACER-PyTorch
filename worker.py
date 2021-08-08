from model import Actor, SDNCritic
import gym
import numpy as np
import torch
from torch import from_numpy
from memory import Memory
from torch.nn.functional import relu
from torch.distributions.multivariate_normal import MultivariateNormal


class Worker:
    def __init__(self,
                 id,
                 n_states,
                 n_actions,
                 env_name,
                 n_hiddens,
                 actions_bounds,
                 global_actor,
                 avg_actor,
                 global_critic,
                 shared_actor_optimizer,
                 shared_critic_optimizer,
                 gamma,
                 ent_coeff,
                 mem_size,
                 k,
                 c,
                 n_sdn,
                 delta,
                 replay_ratio,
                 polyak_coeff):
        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.actions_bounds = actions_bounds
        self.n_hiddens = n_hiddens
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.env_name = env_name
        self.mem_size = mem_size
        self.k = k
        self.c = c
        self.n_sdn = n_sdn
        self.delta = delta
        self.replay_ratio = replay_ratio
        self.polyak_coeff = polyak_coeff
        self.memory = Memory(self.mem_size)
        self.env = gym.make(self.env_name)
        self.env.seed(self.id + 1)

        self.local_actor = Actor(self.n_states, self.n_actions, self.n_hiddens * 2)
        self.local_critic = SDNCritic(self.n_states, self.n_actions)

        self.global_actor = global_actor
        self.avg_actor = avg_actor
        self.global_critic = global_critic
        self.shared_actor_optimizer = shared_actor_optimizer
        self.shared_critic_optimizer = shared_critic_optimizer

        self.mse_loss = torch.nn.MSELoss()
        self.episode = 0
        self.eps = 1e-6

    def get_action(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            dist, mu = self.local_actor(state)
            action = dist.sample()
        action = np.clip(action.numpy(), *self.actions_bounds)
        return action, mu.numpy()

    def get_value(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            q_value, value = self.local_critic(state)
        return q_value.numpy(), value.numpy()

    def sync_thread_spec_params(self, lock):
        with lock:
            self.local_actor.load_state_dict(self.global_actor.state_dict())
            self.local_critic.load_state_dict(self.global_critic.state_dict())

    @staticmethod
    def share_grads_to_global_models(local_model, global_model, lock):
        with lock:
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                # if global_param.grad is not None:
                #     return

                global_param._grad = local_param.grad

    def q_retrace(self, rewards, dones, q_values, values, next_value, rho_i):
        q = next_value
        q_returns = []
        for i in reversed(range(self.k)):
            q = rewards[i] + self.gamma * (~dones[i]) * q
            q_returns.insert(0, q)
            q = rho_i[i] * (q - q_values[i]) + values[i]

        return torch.cat(q_returns).view(-1, 1)

    def soft_update_avg_network(self, lock):
        with lock:
            for avg_param, global_param in zip(self.avg_actor.parameters(), self.global_actor.parameters()):
                avg_param.data.copy_(self.polyak_coeff * global_param.data + (1 - self.polyak_coeff) * avg_param.data)

    def step(self, lock):
        print(f"Worker: {self.id} started.")
        running_reward = 0
        state = self.env.reset()
        next_state = None
        episode_reward = 0
        lock = lock
        while True:
            with lock:
                self.shared_actor_optimizer.zero_grad()  # Reset global gradients
                self.shared_critic_optimizer.zero_grad()
            self.sync_thread_spec_params(lock)  # Synchronize thread-specific parameters

            states, actions, rewards, dones, mus = [], [], [], [], []
            for step in range(1, 1 + self.k):
                action, mu = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action[0])
                # self.env.render()

                states.append(state)
                actions.append(action[0])
                rewards.append(reward)
                dones.append(done)
                mus.append(mu[0])

                episode_reward += reward
                state = next_state

                if done:
                    state = self.env.reset()
                    self.episode += 1
                    if self.episode == 1:
                        running_reward = episode_reward
                    else:
                        running_reward = 0.99 * running_reward + 0.01 * episode_reward

                    if self.id == 0:
                        print(f"\nW{self.id}| Ep {self.episode}| Re {running_reward:.0f}| Mem len {len(self.memory)}")
                    episode_reward = 0

            trajectory = dict(states=states, actions=actions, rewards=rewards,
                              dones=dones, mus=mus, next_state=next_state)
            self.memory.add(**trajectory)

            if len(self.memory) % 4 == 0:
                with lock:
                    self.shared_actor_optimizer.zero_grad()  # Reset global gradients
                    self.shared_critic_optimizer.zero_grad()
                self.sync_thread_spec_params(lock)  # Synchronize thread-specific parameters

                self.train(*self.memory.sample(), lock)

    def train(self, states, actions, rewards, dones, mus, next_state, lock):
        states = torch.Tensor(states)
        mus = torch.Tensor(mus)
        actions = torch.LongTensor(actions).view(-1, 1)
        next_state = torch.Tensor(next_state)
        rewards = torch.Tensor(rewards)
        dones = torch.BoolTensor(dones)

        dist, _ = self.local_actor(states)
        dist_avg, _ = self.avg_actor(states)
        u = [self.get_action(states.numpy(), batch=True)[0] for _ in range(self.n_sdn)]
        u = from_numpy(np.hstack(u).reshape((-1, 1, self.n_sdn))).float()
        
        q_values, values = self.local_critic(states, actions, u)

        f_i = dist.cdf(actions)
        actions_prime = dist.sample()
        f_i_prime = dist.cdf(actions_prime)

        u = torch.stack(torch.Tensor([self.get_action(states.numpy(), batch=True) for _ in range(self.n_sdn)]), dim=2)
        q_values_prime, _ = self.local_critic(states, actions_prime, u)

        with torch.no_grad():
            rho_i = f_i / (MultivariateNormal(mus, 0.3 * torch.eye(mus.size())).cdf(actions) + self.eps)
            rho_i_prime = f_i_prime / (
                        MultivariateNormal(mus, 0.3 * torch.eye(mus.size())).cdf(actions_prime) + self.eps)

            c_i = torch.min(torch.ones_like(rho_i), rho_i.pow(1 / self.n_actions))

            _, next_value = self.get_value(next_state)

            q_ret = self.q_retrace(rewards, dones, q_values, values, next_value, c_i)
            q_opc = self.q_retrace(rewards, dones, q_values, values, next_value, torch.ones_like(c_i))

        ent = dist.entropy().mean()

        # Truncated Importance Sampling:
        adv = q_opc - values
        logf_i = torch.log(f_i + self.eps)
        gain_f = logf_i * adv * torch.min(self.c * torch.ones_like(rho_i), rho_i)
        loss_f = -gain_f.mean()

        # Bias correction for the truncation
        adv_bc = q_values_prime.detach() - values

        logf_bc = torch.log(f_i_prime + 1e-6)
        gain_bc = torch.sum(logf_bc * adv_bc * relu(1 - self.c / (rho_i_prime + self.eps)), dim=-1)
        loss_bc = -gain_bc.mean()

        policy_loss = loss_f + loss_bc
        loss_q = (q_ret - q_values) * q_values + torch.min(torch.ones_like(rho_i), rho_i) * (q_ret - q_values) * values
        loss_q = -loss_q.mean()

        # # trust region:
        g = torch.autograd.grad(-(policy_loss - self.ent_coeff * ent), dist.mean)[0]
        k = -dist_avg.cdf(actions) / (f_i_prime.detach() + 1e-6)
        k_dot_g = torch.sum(k * g, dim=-1, keepdim=True)

        adj = torch.max(torch.zeros_like(k_dot_g),
                        (k_dot_g - self.delta) / (torch.sum(k.square(), dim=-1, keepdim=True) + 1e-6))

        grads_f = -(g - adj * k)
        dist.mean.backward(grads_f)
        # loss = policy_loss - self.ent_coeff * ent
        # loss.backward()
        loss_q.backward()

        self.share_grads_to_global_models(self.local_actor, self.global_actor, lock)
        self.share_grads_to_global_models(self.local_critic, self.global_critic, lock)

        with lock:
            self.shared_actor_optimizer.step()
            self.shared_critic_optimizer.step()

        self.soft_update_avg_network(lock)
