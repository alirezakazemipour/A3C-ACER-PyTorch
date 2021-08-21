from collections import deque
import numpy as np


class Memory:
    def __init__(self, memory_size, seed):
        self.memory_size = memory_size
        np.random.seed(seed)
        self.state_buffer = deque(maxlen=self.memory_size)
        self.action_buffer = deque(maxlen=self.memory_size)
        self.reward_buffer = deque(maxlen=self.memory_size)
        self.mu_buffer = deque(maxlen=self.memory_size)
        self.done_buffer = deque(maxlen=self.memory_size)
        self.next_state_buffer = deque(maxlen=self.memory_size)

    def add(self, **trajectory):
        self.state_buffer.append(trajectory["states"])
        self.action_buffer.append(trajectory["actions"])
        self.reward_buffer.append(trajectory["rewards"])
        self.mu_buffer.append(trajectory["mus"])
        self.done_buffer.append(trajectory["dones"])
        self.next_state_buffer.append(trajectory["next_state"])

        assert len(self.state_buffer) <= self.memory_size

    def sample(self):
        idx = np.random.randint(0, len(self))
        return self.state_buffer[idx], self.action_buffer[idx], self.reward_buffer[idx],\
               self.done_buffer[idx], self.mu_buffer[idx], self.next_state_buffer[idx]

    def __len__(self):
        return len(self.state_buffer)
