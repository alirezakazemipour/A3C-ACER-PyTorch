from NN import Model
import numpy as np
import torch
from torch import from_numpy
from .memory import Memory
from torch.nn.functional import relu
from Utils import make_state, make_atari


class Worker(torch.multiprocessing.Process):
    def __init__(self,
                 id,
                 global_model,
                 avg_model,
                 shared_optimizer,
                 lock,
                 logger,
                 **config
                 ):
        super(Worker, self).__init__()
        self.id = id
        self.config = config
        self.logger = logger
        self.seed = self.config["seed"] + self.id
        self.memory = Memory(self.config["mem_size"], seed=self.seed)
        self.env = make_atari(self.config["env_name"], seed=self.seed)
        self.tau = self.config["polyak_coeff"]

        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        self.local_model = Model(self.config["state_shape"], self.config["n_actions"])
        self.mse_loss = torch.nn.MSELoss()

        self.global_model = global_model
        self.avg_model = avg_model
        self.shared_optimizer = shared_optimizer
        self.lock = lock

        self.eps = 1e-6
        self.episode = 0
        self.iter = 0
        self.step = 0

    def get_actions_and_qvalues(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).byte()
        with torch.no_grad():
            dist, q_values, probs = self.local_model(state)
            action = dist.sample()
        return action.numpy(), q_values.numpy(), probs.numpy()

    def sync_thread_spec_params(self):
        with self.lock:
            self.local_model.load_state_dict(self.global_model.state_dict())

    def update_shared_model(self, grads, model):
        with self.lock:
            for gradient, param in zip(grads, model.parameters()):
                param._grad = gradient

            self.shared_optimizer.step()
            for avg_param, global_param in zip(self.avg_model.parameters(), self.global_model.parameters()):
                avg_param.data.copy_(self.tau * global_param.data + (1 - self.tau) * avg_param.data)

    def q_retrace(self, rewards, dones, q_values, values, next_value, rho_i):
        q_ret = next_value
        q_returns = []
        rho_bar_i = torch.min(torch.ones_like(rho_i), rho_i)
        for i in reversed(range(self.config["k"])):
            q_ret = rewards[i] + self.config["gamma"] * (~dones[i]) * q_ret
            q_returns.insert(0, q_ret)
            q_ret = rho_bar_i[i] * (q_ret - q_values[i]) + values[i]

        return torch.cat(q_returns).view(-1, 1)

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

            rho = f / (mus + self.eps)
            rho_i = rho.gather(-1, actions)

            _, next_q_value, next_mu = self.get_actions_and_qvalues(next_state)
            next_value = (next_q_value * next_mu).sum(-1)

            q_ret = self.q_retrace(rewards, dones, q_i, values, next_value, rho_i)

        ent = dist.entropy().mean()

        # Truncated Importance Sampling:
        adv = q_ret - values
        f_i = f.gather(-1, actions)
        logf_i = torch.log(f_i + self.eps)
        gain_f = logf_i * adv * torch.min(self.config["c"] * torch.ones_like(rho_i), rho_i)
        loss_f = -gain_f.mean()

        # Bias correction for the truncation
        adv_bc = q_values.detach() - values

        logf_bc = torch.log(f + self.eps)
        gain_bc = torch.sum(logf_bc * adv_bc * relu(1 - self.config["c"] / (rho + self.eps)) * f.detach(), dim=-1)
        loss_bc = -gain_bc.mean()

        policy_loss = loss_f + loss_bc
        loss_q = self.config["critic_loss_coeff"] * self.mse_loss(q_ret, q_i)

        # trust region:
        g = torch.autograd.grad(-(policy_loss - self.config["ent_coeff"] * ent), f)[0]
        k = -f_avg / (f.detach() + self.eps)
        k_dot_g = torch.sum(k * g, dim=-1, keepdim=True)

        adj = torch.max(torch.zeros_like(k_dot_g),
                        (k_dot_g - self.config["delta"]) / (torch.sum(k.square(), dim=-1, keepdim=True) + self.eps))

        grads_f = -(g - adj * k)
        f.backward(grads_f, retain_graph=True)
        loss_q.backward()

        grads = [param.grad for param in self.local_model.parameters()]
        self.update_shared_model(grads, self.global_model)
        return (f * grads_f).mean().item(), loss_q.item()

    def run(self):
        print(f"Worker: {self.id} started.")
        state = np.zeros(self.config["state_shape"], dtype=np.uint8)
        obs = self.env.reset()
        state = make_state(state, obs, True)
        next_state = None
        episode_reward = 0
        while True:
            self.iter += 1
            self.shared_optimizer.zero_grad()
            self.sync_thread_spec_params()  # Synchronize thread-specific parameters

            states, actions, rewards, dones, mus = [], [], [], [], []
            for _ in range(1, 1 + self.config["k"]):
                self.step += 1
                print(self.step)
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
                    self.episode += 1
                    with self.lock:
                        self.logger.episodic_log(self.id, self.episode, episode_reward, len(self.memory), self.step)
                    episode_reward = 0
                    self.step = 0

            trajectory = dict(states=states, actions=actions, rewards=rewards,
                              dones=dones, mus=mus, next_state=next_state)
            self.memory.add(**trajectory)
            policy_loss, value_loss = self.train(states, actions, rewards, dones, mus, next_state)
            with self.lock:
                self.logger.training_log(self.id,
                                         self.iter,
                                         policy_loss,
                                         value_loss,
                                         self.global_model,
                                         self.avg_model,
                                         self.shared_optimizer)

            n = np.random.poisson(self.config["replay_ratio"])
            pl, vl = [], []
            for _ in range(n):
                self.shared_optimizer.zero_grad()
                self.sync_thread_spec_params()  # Synchronize thread-specific parameters

                policy_loss, value_loss = self.train(*self.memory.sample())
                pl.append(policy_loss)
                vl.append(value_loss)
            with self.lock:
                self.logger.training_log(self.id,
                                         self.iter,
                                         sum(pl) / n,
                                         sum(vl) / n,
                                         self.global_model,
                                         self.avg_model,
                                         self.shared_optimizer)

