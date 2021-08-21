from comet_ml import Experiment
import gym
from NN import Model, SharedAdam
from Agent import Worker
from Utils import Logger
from torch import multiprocessing as mp
import torch
import os
import yaml
import argparse

# TODOs:
# Add docker support
# Add CircleCI

if __name__ == "__main__":
    with open("training_configs.yml") as f:
        params = yaml.load(f.read(), Loader=yaml.FullLoader)

    params.update({"n_workers": os.cpu_count()})
    params.update({"mem_size": params["total_memory_size"] // params["n_workers"] // params["k"]})
    if not isinstance(params["state_shape"], tuple):
        params["state_shape"] = tuple(params["state_shape"])

    parser = argparse.ArgumentParser(
        description="Variable parameters based on the configuration of the machine or user's choice")
    parser.add_argument("--env_name", default="PongNoFrameskip-v4", type=str, help="Name of the environment.")
    parser.add_argument("--interval", default=50, type=int,
                        help="The interval specifies how often different parameters should be saved and printed,"
                             " counted by episodes.")
    parser.add_argument("--do_train", action="store_true",
                        help="The flag determines whether to train the agent or play with it.")
    parser.add_argument("--seed", default=123, type=int,
                        help="The randomness' seed for torch, numpy, random & gym[env].")
    parser_params = parser.parse_args()

    params = {**vars(parser_params), **params}
    del parser_params

    test_env = gym.make(params["env_name"])
    params.update({"n_actions": test_env.action_space.n})
    params.update({"max_episode_steps": test_env.spec.max_episode_steps})
    test_env.close()
    del test_env

    print(params)

    torch.manual_seed(params["seed"])

    global_model = Model(params["state_shape"], params["n_actions"])
    global_model.share_memory()

    shared_opt = SharedAdam(global_model.parameters(), lr=params["lr"])
    shared_opt.share_memory()

    avg_model = Model(params["state_shape"], params["n_actions"])
    avg_model.load_state_dict(global_model.state_dict())
    avg_model.share_memory()
    for p in avg_model.parameters():
        p.requires_grad = False

    os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy uses only one thread for each process
    os.environ["CUDA_VISABLE_DEVICES"] = ""  # make sure not to use gpu

    mp.set_start_method("spawn")
    lock = mp.Lock()

    experiment = Experiment()
    logger = Logger(experiment=experiment, **params)
    workers = [Worker(id=i,
                      global_model=global_model,
                      avg_model=avg_model,
                      shared_optimizer=shared_opt,
                      lock=lock,
                      logger=logger,
                      **params) for i in range(params["n_workers"])
               ]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
