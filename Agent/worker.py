from NN import Model
import numpy as np
import torch
from torch import from_numpy
from .memory import Memory
from torch.nn.functional import relu
from Utils import make_atari, make_state
import Utils as utils


class Worker(torch.multiprocessing.Process):
    def __init__(self,
                 id,
                 global_model,
                 avg_model,
                 shared_optimizer,
                 **config
                 ):
        super(Worker, self).__init__()
        self.id = id
        self.config = config
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

        self.eps = 1e-6
        self.episode = 0
        self.iter = 0
        self.step = 0

        self.episode_stats = [dict(episode=0,
                                   max_reward=-np.inf,
                                   running_reward=0,
                                   episode_len=0,
                                   mem_len=0
                                   ) for i in range(self.config["n_workers"])
                              ]
        self.iter_stats = [dict(iteration=0,
                                running_ploss=0,
                                running_vloss=0,
                                running_grad_norm=0,
                                np_rng_state=None,
                                env_rng_state=None
                                ) for i in range(self.config["n_workers"])
                           ]

    def set_rng_state(self, *rng_state):
        np.random.set_state(rng_state[0])
        self.memory.set_rng_state(rng_state[1])
        self.env.set_rng_state(*rng_state[2])

    def get_actions_and_qvalues(self, state):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).byte()
        with torch.no_grad():
            dist, q_values, probs = self.local_model(state)
            action = dist.sample()
        return action.numpy(), q_values.numpy(), probs.numpy()

    def sync_thread_spec_params(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    @staticmethod
    def share_grads_to_global_models(local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

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
        next_state = torch.ByteTensor(next_state)
        rewards = torch.CharTensor(rewards)
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

        # grads = [param.grad for param in self.local_model.parameters()]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.config["max_grad_norm"])
        self.share_grads_to_global_models(self.local_model, self.global_model)
        self.shared_optimizer.step()

        for avg_param, global_param in zip(self.avg_model.parameters(), self.global_model.parameters()):
            avg_param.data.copy_(self.tau * global_param.data + (1 - self.tau) * avg_param.data)

        return (f * grads_f).mean().item(), loss_q.item(), grad_norm.item()

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
                action, _, mu = self.get_actions_and_qvalues(state)
                next_obs, reward, done, _ = self.env.step(action[0])
                # self.env.render()

                states.append(state)
                actions.append(action[0].astype(np.uint8))
                rewards.append(np.sign(reward).astype(np.int8))
                dones.append(done)
                mus.append(mu[0])

                episode_reward += reward
                next_state = make_state(state, next_obs, False)
                state = next_state

                if done:
                    obs = self.env.reset()
                    state = make_state(state, obs, True)
                    self.episode += 1
                    self.episode_stats = utils.episodic_log(self.episode_stats,
                                                            self.episode,
                                                            episode_reward,
                                                            len(self.memory),
                                                            self.step)
                    episode_reward = 0
                    self.step = 0

            trajectory = dict(states=states, actions=actions, rewards=rewards,
                              dones=dones, mus=mus, next_state=next_state)
            self.memory.add(**trajectory)
            policy_loss, value_loss, grad_norm = self.train(states, actions, rewards, dones, mus, next_state)
            self.iter_stats = utils.training_log(self.iter_stats,
                                                 self.episode_stats,
                                                 self.id,
                                                 self.iter,
                                                 policy_loss,
                                                 value_loss,
                                                 grad_norm,
                                                 self.global_model,
                                                 self.avg_model,
                                                 self.shared_optimizer,
                                                 np_rng_state=np.random.get_state(),
                                                 mem_rng_state=self.memory.get_rng_state(),
                                                 env_rng_state=self.env.get_rng_state(),
                                                 on_policy=True,
                                                 **self.config)

            n = np.random.poisson(self.config["replay_ratio"])
            pl, vl, g_norm = [], [], []
            for _ in range(n):
                self.shared_optimizer.zero_grad()
                self.sync_thread_spec_params()  # Synchronize thread-specific parameters

                policy_loss, value_loss, grad_norm = self.train(*self.memory.sample())
                pl.append(policy_loss)
                vl.append(value_loss)
                g_norm.append(grad_norm)

                self.iter_stats = utils.training_log(self.iter_stats, None, None, self.iter,
                                                     sum(pl) / len(pl) if n != 0 else 0,
                                                     sum(vl) / len(vl) if n != 0 else 0,
                                                     sum(g_norm) / len(g_norm) if n != 0 else 0,
                                                     None, None, None
                                                     )
