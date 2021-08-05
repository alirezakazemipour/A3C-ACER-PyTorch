import numpy as np
import gym


def make_atari(env_id, episodic_life=True):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    return env


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()

        noops = np.random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)

        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self.skip = skip

    def step(self, action):
        reward = 0
        done = None
        for i in range(self.skip):
            obs, r, done, info = self.env.step(action)

            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            reward += r
            if done:
                break

        max_frame = self.obs_buffer.max(axis=0)

        return max_frame, reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.real_done = done

        lives = info["lives"]
        if self.lives > lives > 0:
            done = True

        self.lives = lives
        return obs, reward, done, info

    def reset(self):

        if self.real_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs
