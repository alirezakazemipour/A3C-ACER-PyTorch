import gym
from model import Actor, SDNCritic
from shared_optimizer import SharedAdam
from worker import Worker
from torch import multiprocessing as mp
import os
import torch
import numpy as np
import random

env_name = "Pendulum-v0"
n_workers = 4
lr = 5e-4
gamma = 0.9
ent_coeff = 0.0001
n_hiddens = 256
mem_size = 5000
k = 50
c = 5
delta = 1
replay_ratio = 4
polyak_coeff = 0.01
n_sdn = 5


def update_shared_model(queue, lock, actor_opt, critic_opt, actor, critic, avg_actor):
    while True:
        actor_grads, critic_grads, id = queue.get()
        # print(f"grads of worker:{id}")
        with lock:
            actor_opt.zero_grad()
            critic_opt.zero_grad()
            for a_grad, param in zip(actor_grads, actor.parameters()):
                param._grad = a_grad
            for c_grad, param in zip(critic_grads, critic.parameters()):
                param._grad = c_grad

            actor_opt.step()
            critic_opt.step()
            for avg_param, global_param in zip(avg_actor.parameters(), actor.parameters()):
                avg_param.data.copy_(polyak_coeff * global_param.data + (1 - polyak_coeff) * avg_param.data)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    os.environ["OMP_NUM_THREADS"] = "1"  # make sure numpy uses only one thread for each process
    os.environ["CUDA_VISABLE_DEVICES"] = ""  # make sure not to use gpu

    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    test_env = gym.make(env_name)
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    actions_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    test_env.close()
    print(f"Env: {env_name}\n"
          f"n_states: {n_states}\n"
          f"n_actions: {n_actions}\n"
          f"n_workers: {n_workers}\n")

    global_actor = Actor(n_states, n_actions, actions_bounds, n_hiddens)
    global_actor.share_memory()

    global_critic = SDNCritic(n_states, n_actions)
    global_critic.share_memory()

    shared_actor_opt = SharedAdam(global_actor.parameters(), lr=lr)
    shared_actor_opt.share_memory()

    shared_critic_opt = SharedAdam(global_critic.parameters(), lr=lr)
    shared_critic_opt.share_memory()

    avg_actor = Actor(n_states, n_actions, actions_bounds, n_hiddens)
    avg_actor.load_state_dict(global_actor.state_dict())
    avg_actor.share_memory()
    for p in avg_actor.parameters():
        p.requires_grad = False

    grad_updates_queue = mp.Queue()
    lock = mp.Lock()

    optimizer_worker = mp.Process(target=update_shared_model,
                                  args=(grad_updates_queue, lock, shared_actor_opt,
                                        shared_critic_opt, global_actor, global_critic, avg_actor))
    optimizer_worker.start()

    workers = [Worker(id=i,
                      n_states=n_states,
                      n_actions=n_actions,
                      actions_bounds=actions_bounds,
                      env_name=env_name,
                      n_hiddens=n_hiddens,
                      global_actor=global_actor,
                      avg_actor=avg_actor,
                      global_critic=global_critic,
                      queue=grad_updates_queue,
                      gamma=gamma,
                      ent_coeff=ent_coeff,
                      mem_size=mem_size,
                      k=k,
                      c=c,
                      n_sdn=n_sdn,
                      delta=delta,
                      replay_ratio=replay_ratio,
                      lock=lock) for i in range(n_workers)]

    processes = []
    for worker in workers:
        worker.start()
        processes.append(worker)

    for p in processes:
        p.join()
