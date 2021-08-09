import gym
from model import Actor, Critic
from shared_optimizer import SharedAdam
from worker import Worker
from torch import multiprocessing as mp
import mujoco_py

env_name = "Walker2d-v2"
n_workers = 6
lr = 5e-4
gamma = 0.9
ent_coeff = 1e-4
n_hiddens = 256


def run_workers(worker, lock):
    worker.step(lock)


def update_shared_model(queue, lock, actor_opt, critic_opt, actor, critic):
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


if __name__ == "__main__":
    test_env = gym.make(env_name)
    n_states = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.shape[0]
    actions_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    test_env.close()
    print(f"Env: {env_name}\n"
          f"n_states: {n_states}\n"
          f"n_actions: {n_actions}\n"
          f"n_workers: {n_workers}\n"
          f"action_bounds: {actions_bounds}")

    global_actor = Actor(n_states, n_actions, actions_bounds, n_hiddens)
    global_actor.share_memory()

    global_critic = Critic(n_states)
    global_critic.share_memory()

    shared_actor_opt = SharedAdam(global_actor.parameters(), lr=lr)
    shared_actor_opt.share_memory()

    shared_critic_opt = SharedAdam(global_critic.parameters(), lr=lr)
    shared_critic_opt.share_memory()

    grad_updates_queue = mp.Queue()
    lock = mp.Lock()

    optimizer_worker = mp.Process(target=update_shared_model,
                                  args=(grad_updates_queue, lock, shared_actor_opt,
                                        shared_critic_opt, global_actor, global_critic))
    optimizer_worker.start()

    workers = [Worker(id=i,
                      n_states=n_states,
                      n_actions=n_actions,
                      action_bounds=actions_bounds,
                      env_name=env_name,
                      n_hiddens=n_hiddens,
                      global_actor=global_actor,
                      global_critic=global_critic,
                      queue=grad_updates_queue,
                      gamma=gamma,
                      ent_coeff=ent_coeff,
                      lock=lock) for i in range(n_workers)
               ]
    processes = []

    for worker in workers:
        worker.start()
        processes.append(worker)

    for p in processes:
        p.join()
