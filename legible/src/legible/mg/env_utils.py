import operator
from functools import reduce

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from minigrid.wrappers import DirectionObsWrapper

from mg.envs import ZigZagEnv, WallsAndLava


class MgFlattenObservation(ObservationWrapper):
    def __init__(self, env, view_size=7, has_goal_dir = True, normalize_direction = True):
        super().__init__(env)
        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)
        self.normalize_direction = normalize_direction
        flat_size = imgSize + 1
        lows = np.zeros((flat_size,))
        highs = np.ones((flat_size,))
        highs = highs * 11 # largest index of tile type
        if not self.normalize_direction:
            lows[-1] = - np.pi / 2
            highs[-1] = np.pi / 2

        self.observation_space = spaces.Box(
            low=lows,
            high=highs,
            shape=(imgSize + 1,),
            dtype=np.float32,
        )

    def observation(self, obs: ObsType, **kwargs):
        if self.normalize_direction:
            gd = np.array([(obs["goal_direction"] + np.pi/2) / np.pi])
        else:
            gd = np.array([obs["goal_direction"]])

        return np.concatenate((obs["image"].flatten(),gd))


def create_mg_env(env_name, render_mode ="rgb_array"):
    if "Zigzag" in env_name:
        zigzags = int(env_name.replace("MiniGrid-Zigzag-",""))
        loc_env = ZigZagEnv(zigzags,render_mode=render_mode)
    elif "walls-and-lava" in env_name:
        [rows,cols] = env_name.replace("MiniGrid-walls-and-lava-","").split("-")
        rows = int(rows)
        cols = int(cols)
        loc_env = WallsAndLava(rows=rows,columns=cols,render_mode=render_mode)
    elif "LavaCrossing" in env_name:
        loc_env = gym.make(env_name, render_mode=render_mode)
        loc_env.action_space = Discrete(3)
    else:
        loc_env = gym.make(env_name, render_mode=render_mode)
    loc_env = DirectionObsWrapper(loc_env, type="angle")
    loc_env = MgFlattenObservation(loc_env)
    return loc_env
