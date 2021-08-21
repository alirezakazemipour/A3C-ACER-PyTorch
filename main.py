import gym
from model import Model
from shared_optimizer import SharedAdam
from worker import Worker
from torch import multiprocessing as mp
import os
import yaml
import argparse


if __name__ == "__main__":
    with open("training_configs.yml") as f:
        params = yaml.load(f.read())

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

    test_env = gym.make(env_name)
    n_actions = test_env.action_space.n
    max_episode_steps = test_env.spec.max_episode_steps
    test_env.close()
    print(f"Env: {env_name}\n"
          f"n_actions: {n_actions}\n"
          f"n_workers: {n_workers}\n")

    mp.set_start_method("spawn")

    os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy uses only one thread for each process
    os.environ["CUDA_VISABLE_DEVICES"] = ""  # make sure not to use gpu

    global_model = Model(state_shape, n_actions)
    global_model.share_memory()

    shared_opt = SharedAdam(global_model.parameters(), lr=lr)
    shared_opt.share_memory()

    avg_model = Model(state_shape, n_actions)
    avg_model.load_state_dict(global_model.state_dict())
    avg_model.share_memory()
    for p in avg_model.parameters():
        p.requires_grad = False

    lock = mp.Lock()
    workers = [Worker(id=i,
                      state_shape=state_shape,
                      n_actions=n_actions,
                      env_name=env_name,
                      global_model=global_model,
                      avg_model=avg_model,
                      shared_optimizer=shared_opt,
                      gamma=gamma,
                      ent_coeff=ent_coeff,
                      mem_size=mem_size,
                      k=k,
                      c=c,
                      delta=delta,
                      lock=lock,
                      replay_ratio=replay_ratio,
                      polyak_coeff=polyak_coeff,
                      critic_coeff=critic_coeff,
                      max_episode_steps=max_episode_steps,
                      ) for i in range(n_workers)
               ]
    processes = []
    for worker in workers:
        worker.start()
        processes.append(worker)

    for p in processes:
        p.join()
