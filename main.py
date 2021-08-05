import gym
from model import Model
from shared_optimizer import SharedRMSProp
# from torch.optim.lr_scheduler import LambdaLR
from worker import Worker
from torch import multiprocessing as mp
import os

env_name = "PongNoFrameskip-v4"
n_workers = os.cpu_count()
lr = 1e-4
gamma = 0.99
update_period = 5
ent_coeff = 0.01
state_shape = (4, 84, 84)
total_episodes = 2000


def run_workers(worker):
    worker.step()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy uses only one thread for each process
    os.environ["CUDA_VISABLE_DEVICES"] = ""  # make sure not to use gpu

    test_env = gym.make(env_name)
    n_actions = test_env.action_space.n
    max_steps_per_episode = test_env.spec.max_episode_steps
    test_env.close()
    print(f"Env: {env_name}\n"
          f"n_actions: {n_actions}\n"
          f"n_workers: {n_workers}\n")

    global_model = Model(state_shape, n_actions)
    global_model.share_memory()

    shared_opt = SharedRMSProp(global_model.parameters(), lr=lr)
    shared_opt.share_memory()

    # schedule_fn = lambda episode: max(1.0 - float(episode / total_episodes), 0)
    # scheduler = LambdaLR(shared_opt, lr_lambda=schedule_fn)

    workers = [Worker(id=i,
                      state_shape=state_shape,
                      n_actions=n_actions,
                      env_name=env_name,
                      global_model=global_model,
                      shared_optimizer=shared_opt,
                      gamma=gamma,
                      ent_coeff=ent_coeff,
                      update_period=update_period,
                      max_steps_per_episode=max_steps_per_episode) for i in range(n_workers)
               ]
    processes = []

    for worker in workers:
        p = mp.Process(target=run_workers, args=(worker,))
        p.daemon = True
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
