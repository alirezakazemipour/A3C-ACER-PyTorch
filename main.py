import gym
from NN import Actor, Critic, SharedAdam
from Agent import Worker

env_name = "Pendulum-v0"
n_workers = 2
lr = 1e-4
gamma = 0.9
ent_coeff = 1e-4
n_hiddens = 256
max_grad = 40

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

    workers = [Worker(id=i,
                      n_states=n_states,
                      n_actions=n_actions,
                      action_bounds=actions_bounds,
                      env_name=env_name,
                      n_hiddens=n_hiddens,
                      global_actor=global_actor,
                      global_critic=global_critic,
                      shared_actor_opt=shared_actor_opt,
                      shared_critic_opt=shared_critic_opt,
                      gamma=gamma,
                      ent_coeff=ent_coeff,
                      max_grad=max_grad) for i in range(n_workers)
               ]

    for worker in workers:
        worker.start()

    for w in workers:
        w.join()
