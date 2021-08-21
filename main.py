import gym
from model import Model
from shared_optimizer import SharedAdam
from worker import Worker
from torch import multiprocessing as mp
import os

env_name = "PongNoFrameskip-v4"
n_workers = os.cpu_count()
lr = 7e-4
gamma = 0.99
ent_coeff = 0.001
k = 20
mem_size = 100000 // n_workers // k
c = 10
delta = 1
replay_ratio = 4
polyak_coeff = 0.01
critic_coeff = 0.5
state_shape = (4, 84, 84)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy uses only one thread for each process
    os.environ["CUDA_VISABLE_DEVICES"] = ""  # make sure not to use gpu

    test_env = gym.make(env_name)
    n_actions = test_env.action_space.n
    max_episode_steps = test_env.spec.max_episode_steps
    test_env.close()
    print(f"Env: {env_name}\n"
          f"n_actions: {n_actions}\n"
          f"n_workers: {n_workers}\n")

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
