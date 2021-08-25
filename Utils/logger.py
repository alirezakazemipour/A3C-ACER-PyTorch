import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import psutil


def init_logger(**config):
    config = config
    log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if config["do_train"] and config["train_from_scratch"]:
        create_wights_folder(log_dir)
        _log_hyperparams(log_dir, **config)
    return log_dir


def create_wights_folder(dir):
    if not os.path.exists("Models"):
        os.mkdir("Models")
    os.mkdir("Models/" + dir)


# region log hyperparameters
def _log_hyperparams(dir, **config):
    with SummaryWriter("Logs/" + dir) as writer:
        for k, v in config.items():
            writer.add_text(k, str(v))
    # endregion


def exp_avg(x, y):
    return 0.99 * x + 0.01 * y


def to_gb(in_bytes):
    return in_bytes / 1024 / 1024 / 1024


def episodic_log(episode_stats, episode, reward, mem_len, episode_len):
    if episode == 1:
        episode_stats["running_reward"] = reward
        episode_stats["episode_len"] = episode_len
    else:
        episode_stats["running_reward"] = exp_avg(episode_stats["running_reward"], reward)
        episode_stats["episode_len"] = exp_avg(episode_stats["episode_len"], episode_len)

    episode_stats["mem_len"] = mem_len
    episode_stats["episode"] = episode
    episode_stats["max_reward"] = max(episode_stats["max_reward"], reward)
    episode_stats["reward"] = reward

    return episode_stats


def training_log(iter_stats,
                 episode_stats,
                 id,
                 iteration,
                 p_loss,
                 v_loss,
                 g_norm,
                 g_model,
                 avg_model,
                 opt,
                 np_rng_state=None,
                 mem_rng_state=None,
                 env_rng_state=None,
                 on_policy=False,
                 **config):
    if iteration == 0:
        iter_stats["running_ploss"] = p_loss
        iter_stats["running_vloss"] = v_loss
        iter_stats["running_grad_norm"] = g_norm
    else:
        iter_stats["running_ploss"] = exp_avg(iter_stats["running_ploss"], p_loss)
        iter_stats["running_vloss"] = exp_avg(iter_stats["running_vloss"], v_loss)
        iter_stats["running_grad_norm"] = exp_avg(iter_stats["running_grad_norm"], g_norm)

    iter_stats["iteration"] = iteration
    iter_stats["np_rng_state"] = np_rng_state
    iter_stats["mem_rng_state"] = mem_rng_state
    iter_stats["env_rng_state"] = env_rng_state

    if on_policy:
        if iteration % (config["interval"] // 3) == 0:
            save_params(episode_stats, iter_stats, id, config["log_dir"], g_model, avg_model, opt)

        if id == 0:
            with SummaryWriter("Logs/" + config["log_dir"]) as writer:
                writer.add_scalar("Max Reward", episode_stats["max_reward"],
                                  episode_stats["episode"])
                writer.add_scalar("Running Reward", episode_stats["running_reward"],
                                  episode_stats["episode"])
                writer.add_scalar("Episode length", episode_stats["episode_len"],
                                  episode_stats["episode"])
                writer.add_scalar("Running PG Loss", iter_stats["running_ploss"], iteration)
                writer.add_scalar("Running Value Loss", iter_stats["running_vloss"], iteration)
                writer.add_scalar("Running Grad Norm", iter_stats["running_grad_norm"], iteration)

            if iteration % config["interval"] == 0:
                ram = psutil.virtual_memory()

                print("Iter: {}| "
                      "E: {}| "
                      "E_Reward: {:.1f}| "
                      "E_Running_Reward: {:.1f}| "
                      "E_length: {:.1f}| "
                      "Mem_length: {}| "
                      "{:.1f}/{:.1f} GB RAM| "
                      "Time: {} "
                      .format(iteration,
                              episode_stats["episode"],
                              episode_stats["reward"],
                              episode_stats["running_reward"],
                              episode_stats["episode_len"],
                              episode_stats["mem_len"],
                              to_gb(ram.used),
                              to_gb(ram.total),
                              datetime.datetime.now().strftime("%H:%M:%S"),
                              )
                      )
    return iter_stats


def save_params(episode_stats, iter_stats, id, dir, g_model, avg_model, opt):
    torch.save({"episode_stats": episode_stats,
                "iter_stats": iter_stats
                },
               "Models/" + dir + "/" + str(id) + "_params.pth")
    if id == 0:
        torch.save({"global_model_state_dict": g_model.state_dict(),
                    "avg_model_state_dict": avg_model.state_dict(),
                    "shared_optimizer_state_dict": opt.state_dict(),
                    },
                   "Models/" + dir + "/net_weights.pth")


def load_weights(**config):
    model_dir = glob.glob("Models/*")
    model_dir.sort()
    log_dir = model_dir[-1].split(os.sep)[-1]

    checkpoints = []
    for i in range(config["n_workers"]):
        checkpoints.append(torch.load(model_dir[-1] + "/" + str(i) + "_params.pth"))

    episode_stats = [checkpoint["episode_stats"] for checkpoint in checkpoints]
    iter_stats = [checkpoint["iter_stats"] for checkpoint in checkpoints]

    checkpoint = torch.load(model_dir[-1] + "/net_weights.pth")

    a = [(iter_stats[i]["np_rng_state"], iter_stats[i]["mem_rng_state"], iter_stats[i]["env_rng_state"])
         for i in range(config["n_workers"])]

    return checkpoint, episode_stats, iter_stats, a, log_dir
