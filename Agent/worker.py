from NN.model import Model
import numpy as np
import torch
from torch import from_numpy
from Utils import make_atari, make_state
import Utils as utils


class Worker(torch.multiprocessing.Process):
    def __init__(self,
                 id,
                 global_model,
                 shared_optimizer,
                 **config):
        super(Worker, self).__init__()
        self.id = id
        self.config = config
        self.seed = self.config["seed"] + self.id
        self.env = make_atari(self.config["env_name"], seed=self.seed)
        np.random.seed(self.seed)

        torch.manual_seed(self.seed)
        self.local_model = Model(self.config["state_shape"], self.config["n_actions"])

        self.global_model = global_model
        self.shared_optimizer = shared_optimizer
        self.mse_loss = torch.nn.MSELoss()

        self.episode = 0
        self.iter = 0

        self.episode_stats = [dict(episode=0,
                                   max_reward=-np.inf,
                                   running_reward=0,
                                   episode_len=0
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
        self.env.set_rng_state(*rng_state[1])

    def get_actions_and_values(self, state, hx, cx):
        state = np.expand_dims(state, 0)
        state = from_numpy(state).byte()
        with torch.no_grad():
            dist, value, (hx, cx) = self.local_model(state, hx, cx)
            action = dist.sample()
        return action.numpy(), value.numpy().squeeze(), (hx, cx)

    def sync_thread_spec_params(self):
        self.local_model.load_state_dict(self.global_model.state_dict())

    @staticmethod
    def share_grads_to_global_models(local_model, global_model):
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

    def run(self):
        print(f"Worker: {self.id} started.")
        state = np.zeros(self.config["state_shape"], dtype=np.uint8)
        obs = self.env.reset()
        state = make_state(state, obs, True)
        next_state = None
        episode_reward = 0
        episode_len = 0
        hx, cx = torch.zeros((1, 256)), torch.zeros((1, 256))
        while True:
            self.iter += 1
            self.shared_optimizer.zero_grad()  # Reset global gradients
            self.sync_thread_spec_params()  # Synchronize thread-specific parameters

            states, actions, rewards, dones, hxs, cxs = [], [], [], [], [], []
            for step in range(1, 1 + self.config["max_episode_steps"]):
                episode_len += 1
                action, _, (next_hx, next_cx) = self.get_actions_and_values(state, hx, cx)
                next_obs, reward, done, _ = self.env.step(action)
                # self.env.render()
                states.append(state)
                actions.append(action)
                rewards.append(np.sign(reward))
                dones.append(done)
                hxs.append(hx)
                cxs.append(cx)

                episode_reward += reward
                next_state = make_state(state, next_obs, False)
                state = next_state
                hx, cx = next_hx, next_cx

                if done:
                    obs = self.env.reset()
                    state = make_state(state, obs, True)

                    self.episode += 1
                    self.episode_stats = utils.episodic_log(self.episode_stats,
                                                            self.episode,
                                                            episode_reward,
                                                            episode_len)
                    episode_reward = 0
                    episode_len = 0
                    hx, cx = torch.zeros((1, 256)), torch.zeros((1, 256))

                if step % self.config["update_period"] == 0:
                    break

            _, R, _ = self.get_actions_and_values(next_state, next_hx, next_cx)
            returns = []
            for r, d in zip(rewards[::-1], dones[::-1]):
                R = r + self.config["gamma"] * R * (1 - d)
                returns.insert(0, R)

            states = torch.Tensor(states).view(-1, *self.config["state_shape"])
            actions = torch.Tensor(actions)
            returns = torch.Tensor(returns).view(-1, 1)
            hxs = torch.cat(hxs)
            cxs = torch.cat(cxs)

            dist, values, _ = self.local_model(states, hxs, cxs)
            log_probs = dist.log_prob(actions.squeeze(1))
            advs = returns - values

            pg_loss = -(log_probs * advs.squeeze(1).detach()).mean()
            value_loss = self.mse_loss(values, returns)

            total_loss = pg_loss - self.config["ent_coeff"] * dist.entropy().mean() + \
                         self.config["critic_loss_coeff"] * value_loss

            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.config["max_grad_norm"])
            self.share_grads_to_global_models(self.local_model, self.global_model)
            self.shared_optimizer.step()

            self.iter_stats = utils.training_log(self.iter_stats,
                                                 self.episode_stats,
                                                 self.id,
                                                 self.iter,
                                                 pg_loss.item(),
                                                 value_loss.item(),
                                                 grad_norm.item(),
                                                 self.global_model,
                                                 self.shared_optimizer,
                                                 np_rng_state=np.random.get_state(),
                                                 env_rng_state=self.env.get_rng_state(),
                                                 **self.config)

    def prepare_to_play(self):
        self.local_model.load_state_dict(self.global_model.state_dict())
        self.local_model.eval()
