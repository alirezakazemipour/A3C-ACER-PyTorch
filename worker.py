from model import Model
import numpy as np
import torch
from torch import from_numpy
from memory import Memory
from torch.nn.functional import relu
from utils import make_state
from atari_wrappers import make_atari


class Worker(torch.multiprocessing.Process):
    def __init__(self,
                 id,
                 state_shape,
                 n_actions,
                 env_name,
                 global_model,
                 avg_model,
                 shared_optimizer,
                 gamma,
                 ent_coeff,
                 mem_size,
                 k,
                 c,
                 delta,
                 replay_ratio,
                 polyak_coeff,
                 critic_coeff,
                 max_episode_steps,
                 lock
                 ):
        super(Worker, self).__init__()
        self.id = id
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.ent_coeff = ent_coeff
        self.env_name = env_name
        self.mem_size = mem_size
        self.k = k
        self.c = c
        self.delta = delta
        self.replay_ratio = replay_ratio
        self.polyak_coeff = polyak_coeff
        self.critic_coeff = critic_coeff
        self.max_episode_steps = max_episode_steps
        self.memory = Memory(self.mem_size)
        self.env = make_atari(self.env_name)

        self.local_model = Model(self.state_shape, self.n_actions)

        self.global_model = global_model
        self.avg_model = avg_model
        self.shared_optimizer = shared_optimizer
        self.lock = lock

        self.mse_loss = torch.nn.MSELoss()
        self.ep = 0

    def get_actions_and_qvalues(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).byte()
        with torch.no_grad():
            dist, values, probs = self.local_model(state)
            action = dist.sample()
        return action.numpy(), values.numpy(), probs.numpy()

    def sync_thread_spec_params(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    def update_shared_model(self, grads, model):
        with self.lock:
            for gradient, param in zip(grads, model.parameters()):
                param._grad = gradient

            self.shared_optimizer.step()
            for avg_param, global_param in zip(self.avg_model.parameters(), self.global_model.parameters()):
                avg_param.data.copy_(self.polyak_coeff * global_param.data + (1 - self.polyak_coeff) * avg_param.data)

    def run(self):
        print(f"Worker: {self.id} started.")
        running_reward = 0
        state = np.zeros(self.state_shape, dtype=np.uint8)
        obs = self.env.reset()
        state = make_state(state, obs, True)
        next_state = None
        episode_reward = 0
        while True:
            self.shared_optimizer.zero_grad()
            self.sync_thread_spec_params()  # Synchronize thread-specific parameters

            states, actions, rewards, dones, mus = [], [], [], [], []
            for step in range(1, 1 + self.k):
                action, _, mu = self.get_actions_and_qvalues(state)
                next_obs, reward, done, _ = self.env.step(action[0])
                # self.env.render()

                states.append(state)
                actions.append(action[0])
                rewards.append(np.sign(reward))
                dones.append(done)
                mus.append(mu[0])

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
                        running_reward = 0.99 * running_reward + 0.01 * episode_reward

                    print(f"\nW{self.id}| Ep {self.ep}| Re {running_reward:.0f}| Mem len {len(self.memory)}")
                    episode_reward = 0

            trajectory = dict(states=states, actions=actions, rewards=rewards,
                              dones=dones, mus=mus, next_state=next_state)
            self.memory.add(**trajectory)
            self.train(states, actions, rewards, dones, mus, next_state)

            n = np.random.poisson(self.replay_ratio)
            for _ in range(n):
                self.shared_optimizer.zero_grad()
                self.sync_thread_spec_params()  # Synchronize thread-specific parameters

                self.train(*self.memory.sample())

    def train(self, states, actions, rewards, dones, mus, next_state):
        states = torch.ByteTensor(states)
        mus = torch.Tensor(mus)
        actions = torch.LongTensor(actions).view(-1, 1)
        next_state = torch.Tensor(next_state)
        rewards = torch.Tensor(rewards)
        dones = torch.BoolTensor(dones)

        dist, q_values, f = self.local_model(states)
        *_, f_avg = self.avg_model(states)
        q_i = q_values.gather(-1, actions)

        with torch.no_grad():
            values = (q_values * f).sum(-1, keepdims=True)

            rho = f / (mus + 1e-6)
            rho_i = rho.gather(-1, actions)

            _, next_q_value, next_mu = self.get_actions_and_qvalues(next_state)
            next_value = (next_q_value * next_mu).sum(-1)

            q_ret = self.q_retrace(rewards, dones, q_i, values, next_value, rho_i)

        ent = dist.entropy().mean()

        # Truncated Importance Sampling:
        adv = q_ret - values
        f_i = f.gather(-1, actions)
        logf_i = torch.log(f_i + 1e-6)
        gain_f = logf_i * adv * torch.min(self.c * torch.ones_like(rho_i), rho_i)
        loss_f = -gain_f.mean()

        # Bias correction for the truncation
        adv_bc = q_values.detach() - values

        logf_bc = torch.log(f + 1e-6)
        gain_bc = torch.sum(logf_bc * adv_bc * relu(1 - self.c / (rho + 1e-6)) * f.detach(), dim=-1)
        loss_bc = -gain_bc.mean()

        policy_loss = loss_f + loss_bc
        loss_q = self.critic_coeff * self.mse_loss(q_ret, q_i)

        # trust region:
        g = torch.autograd.grad(-(policy_loss - self.ent_coeff * ent), f)[0]
        k = -f_avg / (f.detach() + 1e-6)
        k_dot_g = torch.sum(k * g, dim=-1, keepdim=True)

        adj = torch.max(torch.zeros_like(k_dot_g),
                        (k_dot_g - self.delta) / (torch.sum(k.square(), dim=-1, keepdim=True) + 1e-6))

        grads_f = -(g - adj * k)
        f.backward(grads_f, retain_graph=True)
        loss_q.backward()

        grads = [param.grad for param in self.local_model.parameters()]
        self.update_shared_model(grads, self.global_model)

    def q_retrace(self, rewards, dones, q_values, values, next_value, rho_i):
        q_ret = next_value
        q_returns = []
        rho_bar_i = torch.min(torch.ones_like(rho_i), rho_i)
        for i in reversed(range(self.k)):
            q_ret = rewards[i] + self.gamma * (~dones[i]) * q_ret
            q_returns.insert(0, q_ret)
            q_ret = rho_bar_i[i] * (q_ret - q_values[i]) + values[i]

        return torch.cat(q_returns).view(-1, 1)
