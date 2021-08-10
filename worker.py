from model import Actor, SDNCritic
import gym
import numpy as np
import torch
from torch import from_numpy
from memory import Memory
from torch.nn.functional import relu
from torch.distributions import Normal, kl_divergence  # -> MultivariateNormalDiag


class Worker(torch.multiprocessing.Process):
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
                 queue,
                 gamma,
                 ent_coeff,
                 mem_size,
                 k,
                 c,
                 n_sdn,
                 delta,
                 replay_ratio,
                 lock):
        super(Worker, self).__init__()
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
        self.memory = Memory(self.mem_size)
        self.env = gym.make(self.env_name)
        self.env.seed(self.id + 1)

        self.local_actor = Actor(self.n_states, self.n_actions, self.actions_bounds, self.n_hiddens)
        self.local_critic = SDNCritic(self.n_states, self.n_actions)

        self.actor_opt = torch.optim.Adam(self.local_actor.parameters(), lr=0)
        self.critic_opt = torch.optim.Adam(self.local_critic.parameters(), lr=0)

        self.global_actor = global_actor
        self.avg_actor = avg_actor
        self.global_critic = global_critic
        self.queue = queue
        self.lock = lock

        self.mse_loss = torch.nn.MSELoss()
        self.episode = 0
        self.eps = 1e-6

    def get_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            dist, mu, sigma = self.local_actor(state)
            action = dist.sample()
        action = np.clip(action.numpy(), *self.actions_bounds)
        return action, mu.numpy(), sigma.numpy()

    @staticmethod
    def sync_thread_spec_params(from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data)

    def q_retrace(self, rewards, dones, q_values, values, next_value, rho_i):
        q = next_value
        q_returns = []
        for i in reversed(range(self.k)):
            q = rewards[i] + self.gamma * (~dones[i]) * q
            q_returns.insert(0, q)
            q = rho_i[i] * (q - q_values[i]) + values[i]

        return torch.cat(q_returns).view(-1, 1)

    @staticmethod
    def compute_probs(dist, actions):
        return dist.log_prob(actions).exp().view(-1, 1)

    def run(self):
        print(f"Worker: {self.id} started.")
        running_reward = 0
        state = self.env.reset()
        next_state = None
        episode_reward = 0
        while True:
            with self.lock:
                # Synchronize thread-specific parameters
                self.sync_thread_spec_params(self.global_actor, self.local_actor)
                self.sync_thread_spec_params(self.global_critic, self.local_critic)

            states, actions, rewards, dones, mus, sigmas = [], [], [], [], [], []
            for step in range(1, 1 + self.k):
                action, mu, sigma = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action[0])
                # self.env.render()

                states.append(state)
                actions.append(action[0])
                rewards.append((reward + 8.1) / 8.1)
                dones.append(done)
                mus.append(mu[0])
                sigmas.append(sigma[0])

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
                              dones=dones, mus=mus, sigmas=sigmas, next_state=next_state)
            self.memory.add(**trajectory)

            if len(self.memory) % 4 == 0:
                self.train(*self.memory.sample())

    def train(self, states, actions, rewards, dones, mus, sigmas, next_state):
        states = torch.Tensor(states)
        mus = torch.Tensor(mus)
        sigmas = torch.Tensor(sigmas)
        actions = torch.LongTensor(actions).view(-1, self.n_actions)
        next_state = torch.Tensor(next_state).view(-1, self.n_states)
        rewards = torch.Tensor(rewards)
        dones = torch.BoolTensor(dones)

        dist, *_ = self.local_actor(states)
        dist_avg, *_ = self.avg_actor(states)
        u = [dist.sample() for _ in range(self.n_sdn)]
        u = torch.cat(u, dim=-1)
        q_values, values = self.local_critic(states, actions, u)

        f_i = self.compute_probs(dist, actions)
        actions_prime = dist.sample()
        f_i_prime = self.compute_probs(dist, actions_prime)

        u = [dist.sample() for _ in range(self.n_sdn)]
        u = torch.cat(u, dim=-1)
        q_values_prime, _ = self.local_critic(states, actions_prime, u)

        with torch.no_grad():
            rho_i = f_i / (self.compute_probs(Normal(mus, sigmas), actions) + self.eps)
            rho_i_prime = f_i_prime / (self.compute_probs(Normal(mus, sigmas), actions_prime) + self.eps)

            c_i = torch.min(torch.ones_like(rho_i), rho_i.pow(1 / self.n_actions))

            _, next_value = self.local_critic(next_state, None, None)

            q_ret = self.q_retrace(rewards, dones, q_values, values, next_value, c_i)
            q_opc = self.q_retrace(rewards, dones, q_values, values, next_value, torch.ones_like(c_i))

        ent = dist.entropy().mean()

        # Truncated Importance Sampling:
        adv = q_opc - values.detach()
        logf_i = torch.log(f_i + self.eps)
        gain_f = logf_i * adv * torch.min(self.c * torch.ones_like(rho_i), rho_i)
        loss_f = -gain_f.mean()

        # Bias correction for the truncation
        adv_bc = q_values_prime.detach() - values.detach()

        logf_bc = torch.log(f_i_prime + self.eps)

        gain_bc = logf_bc * adv_bc * relu(1 - self.c / (rho_i_prime + self.eps))
        loss_bc = -gain_bc.mean()

        policy_loss = loss_f + loss_bc
        loss_q = (q_ret - q_values.detach()) * q_values + torch.min(torch.ones_like(rho_i), rho_i) * (
                q_ret - q_values.detach()) * values
        loss_q = -loss_q.mean()

        # trust region:
        g = torch.autograd.grad(-(policy_loss - self.ent_coeff * ent), (dist.mean, dist.stddev))
        # k = kl_divergence(dist_avg, dist)
        k = torch.log(dist.stddev / dist_avg.stddev) - 0.5 + (dist_avg.stddev ** 2 + (dist_avg.mean - dist.mean) ** 2) / 2 / (dist.stddev ** 2)
        k = torch.autograd.grad(k.mean(), (dist.mean, dist.stddev))
        # k = -self.compute_probs(dist_avg, actions) / (f_i_prime.detach() + self.eps)
        grads_f = []
        for g_, k_ in zip(g, k):
            k_dot_g = torch.sum(k_ * g_, dim=-1, keepdim=True)

            adj = torch.max(torch.zeros_like(k_dot_g),
                            (k_dot_g - self.delta) / (torch.sum(k_.square(), dim=-1, keepdim=True) + self.eps))

            grads_f.append(-(g_ - adj * k_))

        # loss_pg = policy_loss - self.ent_coeff * ent

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()

        # loss_pg.backward()
        # grads_f = torch.cat(grads_f, dim=-1)
        torch.autograd.backward((dist.mean, dist.stddev), grads_f)
        loss_q.backward()

        a_grads = [param.grad for param in self.local_actor.parameters()]
        c_grads = [param.grad for param in self.local_critic.parameters()]

        self.queue.put((a_grads, c_grads, self.id))
