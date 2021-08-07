import gym
from model import Actor, Critic
from shared_optimizer import SharedAdam
from worker import Worker
from torch import multiprocessing as mp
import os

env_name = "LunarLander-v2"
n_workers = 4
lr = 1e-4
gamma = 0.99
ent_coeff = 0.001
n_hiddens = 128
mem_size = 5000
k = 20
c = 10
delta = 1
replay_ratio = 4
polyak_coeff = 0.01


def run_workers(worker):
    worker.step()


if __name__ == "__main__":
    mp.set_start_method("spawn")

    os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy uses only one thread for each process
    os.environ["CUDA_VISABLE_DEVICES"] = ""  # make sure not to use gpu

    test_env = gym.make(env_name)
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.n
    test_env.close()
    print(f"Env: {env_name}\n"
          f"n_states: {n_states}\n"
          f"n_actions: {n_actions}\n"
          f"n_workers: {n_workers}\n")

    global_actor = Actor(n_states, n_actions, n_hiddens * 2)
    global_actor.share_memory()

    global_critic = Critic(n_states, n_actions)
    global_critic.share_memory()

    shared_actor_opt = SharedAdam(global_actor.parameters(), lr=lr)
    shared_actor_opt.share_memory()

    shared_critic_opt = SharedAdam(global_critic.parameters(), lr=lr * 10)
    shared_critic_opt.share_memory()

    avg_actor = Actor(n_states, n_actions, n_hiddens * 2)
    avg_actor.load_state_dict(global_actor.state_dict())
    avg_actor.share_memory()
    for p in avg_actor.parameters():
        p.requires_grad = False

    workers = [Worker(id=i,
                      n_states=n_states,
                      n_actions=n_actions,
                      env_name=env_name,
                      n_hiddens=n_hiddens,
                      global_actor=global_actor,
                      avg_actor=avg_actor,
                      global_critic=global_critic,
                      shared_actor_optimizer=shared_actor_opt,
                      shared_critic_optimizer=shared_critic_opt,
                      gamma=gamma,
                      ent_coeff=ent_coeff,
                      mem_size=mem_size,
                      k=20,
                      c=10,
                      delta=delta,
                      replay_ratio=replay_ratio,
                      polyak_coeff=polyak_coeff) for i in range(n_workers)
               ]
    processes = []

    for worker in workers:
        p = mp.Process(target=run_workers, args=(worker,))
        p.daemon = True
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
