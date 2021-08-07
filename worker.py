from model import Actor, Critic
import gym
import numpy as np
import torch
from torch import from_numpy
from memory import Memory
from torch.nn.functional import relu


class Worker:
    def __init__(self,
                 id,
                 n_states,
                 n_actions,
                 env_name,
                 n_hiddens,
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
                 delta,
                 replay_ratio):
        self.id = id
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.env_name = env_name
        self.mem_size = mem_size
        self.k = k
        self.c = c
        self.delta = delta
        self.replay_ratio = replay_ratio
        self.memory = Memory(self.mem_size)
        self.env = gym.make(self.env_name)

        self.local_actor = Actor(self.n_states, self.n_actions, self.n_hiddens * 2)
        self.local_critic = Critic(self.n_states, self.n_actions)

        self.global_actor = global_actor
        self.avg_actor = avg_actor
        self.global_critic = global_critic
        self.shared_actor_optimizer = shared_actor_optimizer
        self.shared_critic_optimizer = shared_critic_optimizer

        self.mse_loss = torch.nn.MSELoss()
        self.ep = 0

    def get_action(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).float()
        with torch.no_grad():
            dist, probs = self.local_actor(state)
            action = dist.sample()
        return action.numpy(), probs.numpy()

    def get_q_value(self, state, action):
        state = np.expand_dims(state, 0)
        action = np.expand_dims(action, 0)
        state = from_numpy(state).float()
        action = from_numpy(action).float()
        with torch.no_grad():
            q_value = self.local_critic(state, action)
        return q_value.numpy()

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

            states, actions, rewards, dones, mus = [], [], [], [], []

            for step in range(1, 1 + self.env.spec.max_episode_steps):
                action, mu = self.get_action(state)
                # print(action)
                next_state, reward, done, _ = self.env.step(action[0])

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                mus.append(mu)

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

                if step % self.k == 0:
                    break

            trajectory = dict(states=states, actions=actions, rewards=rewards,
                              dones=dones, mus=mus, next_state=next_state)
            self.memory.add(**trajectory)

            self.train(states, actions, rewards, dones, mus, next_state)

            n = np.random.poisson(self.replay_ratio)
            for _ in range(n):
                self.train(self.memory.sample(), on_policy=False)

    def train(self, states, actions, rewards, dones, mus, next_state, on_policy=True):
        states = torch.Tensor(states)
        mus = torch.Tensor(mus)
        actions = torch.ByteTensor(actions)
        next_state = torch.Tensor(next_state)
        rewards = torch.Tensor(rewards)
        dones = torch.BoolTensor(dones)

        dist, f = self.local_actor(states)
        _, f_avg = self.avg_actor(states)
        q_values = self.local_critic(states, actions)
        values = (q_values * f.detach()).sum(-1)

        with torch.no_grad():
            if on_policy:
                rho = torch.ones((self.k, self.n_actions))
            else:
                rho = f / mus
            rho_i = rho.gather(-1, actions)

            next_action, next_mu = self.get_action(next_state)
            next_q_value = self.get_q_value(next_state, torch.Tensor(next_action))
            next_value = (next_q_value * next_mu).sum(-1)

            q_ret = self.q_retrace(rewards, dones, q_values, values, next_value, rho_i)

        ent = dist.entropy().mean()

        # Truncated Importance Sampling:
        adv = q_ret - values
        f_i = f.gather(-1, actions)
        logf_i = torch.log(f_i)
        gain_f = logf_i * adv.detach() * torch.min(self.c, rho_i)
        loss_f = -gain_f.mean()

        # Bias correction for the truncation
        adv_bc = q_values - values
        logf_bc = torch.log(f)
        gain_bc = torch.sum(logf_bc * adv_bc.detach() * relu(1 - self.c / rho) * f, dim=-1)
        loss_bc = -gain_bc.mean()

        policy_loss = loss_f + loss_bc
        loss_q = self.mse_loss(q_ret, q_values)

        # trust region:
        g = torch.autograd.grad(- (policy_loss - self.ent_coeff * ent), f)
        k = - f_avg / f.detach()
        k_dot_g = torch.sum(k * g, dim=-1)
        adj = torch.max(0, (k_dot_g - self.delta) / torch.sum(k.square(), dim=-1))

        grads_f = - (g - adj * k)

        (f * grads_f).backward()
        loss_q.backward()

        self.share_grads_to_global_models(self.local_actor, self.global_actor)
        self.share_grads_to_global_models(self.local_critic, self.global_critic)

        self.shared_actor_optimizer.step()
        self.shared_critic_optimizer.step()

        self.soft_update_avg_network()

    def q_retrace(self, rewards, dones, q_values, values, next_value, rho_i):
        q_ret = next_value
        q_returns = []
        rho_bar_i = rho_i.clamp_max(1)
        for r, d, rho_bar, q_value, value in zip(rewards[::-1], dones[::-1], rho_bar_i[::-1], q_values[::-1],
                                                 values[::-1]):
            q_ret = r + self.gamma * (~d) * q_ret
            q_returns.insert(0, q_ret)
            q_ret = rho_bar * (q_ret - q_value) + value

        return torch.cat(q_returns)

    def soft_update_avg_network(self, tau=0.005):
        for avg_param, global_param in zip(self.avg_actor.parameters(), self.global_actor.parameters()):
            avg_param.data.copy_(tau * global_param.data + (1 - tau) * avg_param.data)
