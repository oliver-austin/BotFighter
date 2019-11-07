import numpy as np
import random
from collections import deque

"""This buffer class was adapted from the following tutorial: """

class ExperienceBuffer:
    def __init__(self, size):
        self.buffer = deque()
        self.elements = 0
        self.size = size

    def push_back(self, action, state, reward, end, next_state):
        if self.elements < self.size:
            self.buffer.append((action, state, reward, end, next_state))
            self.elements += 1
        else:
            self.buffer.popleft()
            self.buffer.append((action, state, reward, end, next_state))

    def size(self):
        return self.elements

    def sample(self, size):
        batch_sample = []

        if self.elements < size:
            batch_sample = random.sample(self.buffer, self.elements)
        else:
            batch_sample = random.sample(self.buffer, size)

        batch_action, batch_state, batch_reward, batch_end, batch_next_state = list(map(np.array, list(zip(*batch_sample))))
        return batch_action, batch_state, batch_reward, batch_end, batch_next_state