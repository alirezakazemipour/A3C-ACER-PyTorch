import numpy as np
import datetime
import torch


class Logger:
    def __init__(self, **config):
        self.config = config
        self.log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.experiment = self.config["experiment"]
        self.episode_stats = [dict(episode=0,
                                   max_reward=-np.inf,
                                   running_reward=0,
                                   mem_len=0,
                                   episode_len=0
                                   ) for i in range(self.config["n_workers"])]
        self.iter_stats = [dict(running_ploss=0,
                                running_vloss=0,
                                ) for i in range(self.config["n_workers"])]
        self.exp_avg = lambda x, y: 0.99 * x + 0.01 * y if (y != 0).all() else y

        self._log_hyperparams()

    # region log hyperparameters
    def _log_hyperparams(self):
        self.experiment.log_parameters(self.config)

    # endregion

    def episodic_log(self, id, episode, reward, mem_len, episode_len):
        if episode == 1:
            self.episode_stats[id]["running_reward"] = reward
            self.episode_stats[id]["episode_len"] = episode_len

        else:
            self.episode_stats[id]["running_reward"] = self.exp_avg(self.episode_stats[id]["running_reward"], reward)
            self.episode_stats[id]["episode_len"] = self.exp_avg(self.episode_stats[id]["episode_len"], episode_len)

        self.episode_stats[id]["mem_len"] = mem_len
        self.episode_stats[id]["episode"] = episode
        self.episode_stats[id]["max_reward"] = max(self.episode_stats[id]["max_episode"], reward)

    def training_log(self, id, iteration, p_loss, v_loss, g_model, avg_model, opt):
        if iteration == 0:
            self.iter_stats[id]["running_ploss"] = p_loss
            self.iter_stats[id]["running_vloss"] = v_loss
        else:
            self.iter_stats[id]["running_ploss"] = self.exp_avg(self.iter_stats[id]["running_ploss"], p_loss)
            self.iter_stats[id]["running_vloss"] = self.exp_avg(self.iter_stats[id]["running_vloss"], v_loss)

        if id == 0:

            if iteration % (self.config["interval"] // 3) == 0:
                self.save_params(iteration, g_model, avg_model, opt)

            self.experiment.log_metric("Max Reward", self.episode_stats[id]["max_reward"],
                                       self.episode_stats[id]["episode"])
            self.experiment.log_metric("Running Reward", self.episode_stats[id]["running_reward"],
                                       self.episode_stats[id]["episode"])
            self.experiment.log_metric("Episode length", self.episode_stats[id]["episode_len"],
                                       self.episode_stats[id]["episode"])
            self.experiment.log_metric("Running PG Loss", self.iter_stats[id]["running_ploss"], iteration)
            self.experiment.log_metric("Running Value Loss", self.iter_stats[id]["running_vloss"], iteration)

            if iteration % self.config["interval"] == 0:
                print("Iter: {}| "
                      "E: {}| "
                      "E_Running_Reward: {:.1f}| "
                      "E_length:{}| "
                      "Mem_length:{}| "
                      "Time:{} "
                      .format(iteration,
                              self.episode_stats[id]["episode"],
                              self.episode_stats[id]["running_reward"],
                              self.episode_stats[id]["episode_len"],
                              self.episode_stats[id]["mem_len"],
                              datetime.datetime.now().strftime("%H:%M:%S"),
                              )
                      )

    def save_params(self, iteration, g_model, avg_model, opt):
        torch.save({"global_model_state_dict": g_model.state_dict(),
                    "average_model_state_dict": avg_model.state_dict(),
                    "shared_optimizer_state_dict": opt.state_dict(),
                    "iteration": iteration,
                    "episode_stats": self.episode_stats,
                    "iter_stats": self.iter_stats
                    },
                   "Models/" + self.log_dir + "/params.pth")
