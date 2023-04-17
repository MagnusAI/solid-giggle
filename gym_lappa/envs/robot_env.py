import gym
from gym import spaces
import numpy as np


class RobotEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            np.array[0, 0, 0, 0, 0], np.array[1, 1, 1, 1, 1], dtype=np.int32)

    def reset(self):
        
