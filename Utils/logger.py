import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import psutil


class Logger:
    def __init__(self, **config):
        self.config = config
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.episode_stats = [dict(episode=0,
                                   max_reward=-np.inf,
                                   running_reward=0,
                                   mem_len=0,
                                   episode_len=0
                                   ) for i in range(self.config["n_workers"])
                              ]
        self.iter_stats = [dict(iteration=0,
                                running_ploss=0,
                                running_vloss=0,
                                running_grad_norm=0,
                                np_rng_state=None,
                                mem_rng_state=None,
                                env_rng_state=None
                                ) for i in range(self.config["n_workers"])
                           ]

        if self.config["do_train"] and self.config["train_from_scratch"]:
            self.create_wights_folder(self.log_dir)
            self.log_params()

    @staticmethod
    def create_wights_folder(dir):
        if not os.path.exists("Models"):
            os.mkdir("Models")
        os.mkdir("Models/" + dir)

    def log_params(self):
        with SummaryWriter("Logs/" + self.log_dir) as writer:
            for k, v in self.config.items():
                writer.add_text(k, str(v))

    # region log hyperparameters
    def _log_hyperparams(self):
        with SummaryWriter("Logs/" + self.log_dir) as writer:
            for k, v in self.config.items():
                writer.add_text(k, str(v))

    # endregion
    @staticmethod
    def exp_avg(x, y):
        return 0.99 * x + 0.01 * y

    @staticmethod
    def to_gb(in_bytes):
        return in_bytes/ 1024 / 1024 / 1024

    def episodic_log(self, id, episode, reward, mem_len, episode_len):
        if episode == 1:
            self.episode_stats[id]["running_reward"] = reward
            self.episode_stats[id]["episode_len"] = episode_len

        else:
            self.episode_stats[id]["running_reward"] = self.exp_avg(self.episode_stats[id]["running_reward"], reward)
            self.episode_stats[id]["episode_len"] = self.exp_avg(self.episode_stats[id]["episode_len"], episode_len)

        self.episode_stats[id]["mem_len"] = mem_len
        self.episode_stats[id]["episode"] = episode
        self.episode_stats[id]["max_reward"] = max(self.episode_stats[id]["max_reward"], reward)

    def training_log(self, id,
                     iteration,
                     p_loss,
                     v_loss,
                     g_norm,
                     g_model,
                     avg_model,
                     opt, on_policy=False,
                     np_rng_state=None,
                     mem_rng_state=None,
                     env_rng_state=None):

        if iteration == 0:
            self.iter_stats[id]["running_ploss"] = p_loss
            self.iter_stats[id]["running_vloss"] = v_loss
            self.iter_stats[id]["running_grad_norm"] = g_norm
        else:
            self.iter_stats[id]["running_ploss"] = self.exp_avg(self.iter_stats[id]["running_ploss"], p_loss)
            self.iter_stats[id]["running_vloss"] = self.exp_avg(self.iter_stats[id]["running_vloss"], v_loss)
            self.iter_stats[id]["running_grad_norm"] = self.exp_avg(self.iter_stats[id]["running_grad_norm"], g_norm)

        self.iter_stats[id]["iteration"] = iteration
        self.iter_stats[id]["np_rng_state"] = np_rng_state
        self.iter_stats[id]["mem_rng_state"] = mem_rng_state
        self.iter_stats[id]["env_rng_state"] = env_rng_state

        if id == 0 and on_policy:

            if iteration % (self.config["interval"] // 3) == 0:
                self.save_params(iteration, g_model, avg_model, opt)

            with SummaryWriter("Logs/" + self.log_dir) as writer:
                writer.add_scalar("Max Reward", self.episode_stats[id]["max_reward"],
                                       self.episode_stats[id]["episode"])
                writer.add_scalar("Running Reward", self.episode_stats[id]["running_reward"],
                                       self.episode_stats[id]["episode"])
                writer.add_scalar("Episode length", self.episode_stats[id]["episode_len"],
                                       self.episode_stats[id]["episode"])
                writer.add_scalar("Running PG Loss", self.iter_stats[id]["running_ploss"], iteration)
                writer.add_scalar("Running Value Loss", self.iter_stats[id]["running_vloss"], iteration)
                writer.add_scalar("Running Grad Norm", self.iter_stats[id]["running_grad_norm"], iteration)

            if iteration % self.config["interval"] == 0:
                ram = psutil.virtual_memory()

                print("Iter: {}| "
                      "E: {}| "
                      "E_Running_Reward: {:.1f}| "
                      "E_length:{:.1f}| "
                      "Mem_length:{}| "
                      "{:.1f}/{:.1f} GB RAM| "
                      "Time:{} "
                      .format(iteration,
                              self.episode_stats[id]["episode"],
                              self.episode_stats[id]["running_reward"],
                              self.episode_stats[id]["episode_len"],
                              self.episode_stats[id]["mem_len"],
                              self.to_gb(ram.used),
                              self.to_gb(ram.total),
                              datetime.datetime.now().strftime("%H:%M:%S"),
                              )
                      )

    def save_params(self, iteration, g_model, avg_model, opt):
        torch.save({"global_model_state_dict": g_model.state_dict(),
                    "average_model_state_dict": avg_model.state_dict(),
                    "shared_optimizer_state_dict": opt.state_dict(),
                    "episode_stats": self.episode_stats,
                    "iter_stats": self.iter_stats
                    },
                   "Models/" + self.log_dir + "/params.pth")

    def load_weights(self):
        model_dir = glob.glob("Models/*")
        model_dir.sort()
        checkpoint = torch.load(model_dir[-1] + "/params.pth")
        self.log_dir = model_dir[-1].split(os.sep)[-1]

        self.episode_stats = checkpoint["episode_stats"]
        self.iter_stats = checkpoint["iter_stats"]

        a = [self.iter_stats[i]["np_rng_state"] for i in range(self.config["n_workers"])]
        b = [self.iter_stats[i]["mem_rng_state"] for i in range(self.config["n_workers"])]
        c = [self.iter_stats[i]["env_rng_state"] for i in range(self.config["n_workers"])]

        return checkpoint, [self.episode_stats[i]["episode"] for i in range(self.config["n_workers"])], \
               [self.iter_stats[i]["iteration"] for i in range(self.config["n_workers"])], \
               np.stack([a, b, c], axis=-1)
