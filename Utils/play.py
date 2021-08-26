import gym
import numpy as np
from Utils import make_state, make_atari
import time
import torch


class Play:
    def __init__(self, agent, max_episode=1, **config):
        self.config = config
        self.env = make_atari(self.config["env_name"], episodic_life=False, seed=int(time.time()))
        self.env = gym.wrappers.Monitor(self.env, "./Vid", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.prepare_to_play()

    def evaluate(self):
        state = np.zeros(shape=self.config["state_shape"], dtype=np.uint8)
        total_reward = 0
        print("--------Play mode--------")
        for _ in range(self.max_episode):
            done = False
            obs = self.env.reset()
            episode_reward = 0
            state = make_state(state, obs, True)
            hx, cx = torch.zeros((1, 256)), torch.zeros((1, 256))

            while not done:
                action, _, (next_hx, next_cx) = self.agent.get_actions_and_values(state, hx, cx)
                next_obs, r, done, _ = self.env.step(action[0])
                self.env.render()
                time.sleep(0.01)
                episode_reward += r
                nex_state = make_state(state, next_obs, False)
                state = nex_state
                hx, cx = next_hx, next_cx
            total_reward += episode_reward

        print("Total episode reward:", total_reward / self.max_episode)
        self.env.close()
