import json
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import psutil


# region init_logger
def init_logger(**config):
    config = config
    log_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if config["do_train"] and config["train_from_scratch"]:
        create_folders(log_dir)
        _log_hyperparams(log_dir, **config)
    return log_dir


# endregion


# region create_wights_folder
def create_folders(dir):
    if not os.path.exists("Models"):
        os.mkdir("Models")
    if not os.path.exists("Logs"):
        os.mkdir("Logs")

    os.mkdir("Models/" + dir)
    os.mkdir("Logs/" + dir)


# endregion

# region log hyperparameters
def _log_hyperparams(dir, **config):
    write_to_json(config, log_dir=dir)
    with SummaryWriter("Logs/" + dir + "/events" + "/") as writer:
        for k, v in config.items():
            writer.add_text(k, str(v))


# endregion

# region exp_avg
def exp_avg(x, y):
    return 0.99 * x + 0.01 * y


# endregion

# region to_gb
def to_gb(in_bytes):
    return in_bytes / 1024 / 1024 / 1024


# endregion


def episodic_log(episode_stats, episode, reward, episode_len):
    if episode == 1:
        episode_stats["running_reward"] = reward
        episode_stats["episode_len"] = episode_len

    else:
        episode_stats["running_reward"] = exp_avg(episode_stats["running_reward"], reward)
        episode_stats["episode_len"] = exp_avg(episode_stats["episode_len"], episode_len)

    episode_stats["episode"] = episode
    episode_stats["max_reward"] = max(episode_stats["max_reward"], reward)
    return episode_stats


def training_log(iter_stats,
                 episode_stats,
                 id,
                 iteration,
                 p_loss,
                 v_loss,
                 g_norm,
                 g_model,
                 opt,
                 np_rng_state=None,
                 env_rng_state=None,
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
    iter_stats["env_rng_state"] = env_rng_state

    if iteration % (config["interval"] // 3) == 0:
        save_params(episode_stats, iter_stats, id, config["log_dir"], g_model, opt)

    if id == 0:
        with SummaryWriter("Logs/" + config["log_dir"] + "/events" + "/") as writer:
            writer.add_scalar("Max Reward", episode_stats["max_reward"],
                              episode_stats["episode"])
            writer.add_scalar("Running Reward", episode_stats["running_reward"],
                              episode_stats["episode"])
            writer.add_scalar("Episode length", episode_stats["episode_len"],
                              episode_stats["episode"])
            writer.add_scalar("Running PG Loss", iter_stats["running_ploss"], iteration)
            writer.add_scalar("Running Value Loss", iter_stats["running_vloss"], iteration)
            writer.add_scalar("Running Grad Norm", iter_stats["running_grad_norm"], iteration)

        logs_to_write = {"Max Reward": episode_stats["max_reward"],
                         "Running Reward": episode_stats["running_reward"],
                         "Episode length": episode_stats["episode_len"]
                         }

        write_to_json(logs_to_write, **config)
        write_to_csv(logs_to_write, **config)

        if iteration % config["interval"] == 0:
            ram = psutil.virtual_memory()

            print("Iter: {}| "
                  "E: {}| "
                  "E_Running_Reward: {:.1f}| "
                  "E_length:{:.1f}| "
                  "{:.1f}/{:.1f} GB RAM| "
                  "Time:{} "
                  .format(iteration,
                          episode_stats["episode"],
                          episode_stats["running_reward"],
                          episode_stats["episode_len"],
                          to_gb(ram.used),
                          to_gb(ram.total),
                          datetime.datetime.now().strftime("%H:%M:%S"),
                          )
                  )

            del ram
        del logs_to_write
    return iter_stats


def write_to_json(keys_values, **config):
    path = "Logs/" + config["log_dir"] + "/logs.json"
    assert path.endswith(".json")

    with open(path, "a+") as f:
        f.write(json.dumps(keys_values) + "\n")
        f.flush()


def write_to_csv(keys_values, **config):
    path = "Logs/" + config["log_dir"] + "/logs.csv"
    assert path.endswith(".csv")

    if not os.path.exists(path):
        with open(path, "w") as f:
            for i, k in enumerate(keys_values.keys()):
                if i > 0:
                    f.write(",")
                f.write(k)
            f.write("\n")
            f.flush()

    with open(path, "a+") as f:
        for i, v in enumerate(keys_values.values()):
            if i > 0:
                f.write(",")
            f.write(str(v))
        f.write("\n")
        f.flush()


def save_params(episode_stats, iter_stats, id, dir, g_model, opt):
    torch.save({"episode_stats": episode_stats,
                "iter_stats": iter_stats
                },
               "Models/" + dir + "/" + str(id) + "_params.pth")
    if id == 0:
        torch.save({"global_model_state_dict": g_model.state_dict(),
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

    a = [(iter_stats[i]["np_rng_state"], iter_stats[i]["env_rng_state"]) for i in range(config["n_workers"])]

    del checkpoints
    del model_dir
    return checkpoint, episode_stats, iter_stats, a, log_dir
