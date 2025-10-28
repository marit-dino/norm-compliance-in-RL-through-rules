import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.core import ObsType, WrapperObsType
from typing import Any
from gymnasium import spaces

from .gym_gardener import GardenerEnv, PLAYER, WALL, TARGET


class SimpleStateWrapper(ObservationWrapper):
    def __init__(self,env,width,height):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = Box(low=np.array([0,0]),
                                     high = np.array([width,height]))
    def observation(self, observation: ObsType) -> WrapperObsType:
        o = observation["image"]
        found = False
        for row in range(self.height):
            for col in range(self.width):
                if o[col][row] == PLAYER:
                    found = True
                    break
            if found:
                break
        if not found:
            row = self.height
            col = self.width
        return np.array([row,col])

class OheWrapper(ObservationWrapper):
    def __init__(self,env,image_width, image_height, max_value, values_start_at_one):
        super().__init__(env)
        flat_size = image_width * image_height
        self.max_value = max_value
        self.flat_size = flat_size * max_value + 1
        self.values_start_at_one = values_start_at_one
        lows = np.zeros((self.flat_size,))
        highs = np.ones((self.flat_size,))
        lows[-1] = env.observation_space.low[-1]
        highs[-1] = env.observation_space.high[-1]
        self.observation_space = Box(low=lows, high=highs)

    def observation(self, observation: ObsType) -> WrapperObsType:
        obs = np.zeros((self.flat_size,))
        obs[-1] = observation[-1]
        offset = -1 if self.values_start_at_one else 0
        for i in range(observation.shape[0]-1):
            obs[i*self.max_value + int(observation[i]) - offset] = 1
        return obs

class FlattenObservation(ObservationWrapper):
    def __init__(self, env, normalize_direction = True):
        super().__init__(env)
        imgSpace = self.observation_space.spaces["image"].shape
        imgSize = imgSpace[0]*imgSpace[1]
        self.normalize_direction = normalize_direction
        flat_size = imgSize + 1
        lows = np.zeros((flat_size,))
        highs = np.ones((flat_size,))
        highs = highs * 5 # largest index of tile type
        if not self.normalize_direction:
            lows[-1] = -np.pi
            highs[-1] = np.pi

        self.observation_space = Box(
            low=lows,
            high=highs,
            shape=(imgSize + 1,),
            dtype="float32",
        )
    def observation(self, obs: ObsType, **kwargs):
        return self.observation_param(obs,self.normalize_direction)

    @staticmethod
    def observation_param(obs, normalize_direction):
        if normalize_direction:
            gd = np.array([(obs["goal_direction"] + np.pi) / (2*np.pi)])
        else:
            gd = np.array([obs["goal_direction"]])
        flat_image = [item for sublist in obs["image"] for item in sublist]
        return np.concatenate((flat_image,gd))

class DirectionAngleObsWrapper(ObservationWrapper):
    """
    Modification from DirectionObsWrapper. Originally, arctan() was used, but this leads to a big loss of information!
    Also, there were problems with the division by zero.
    Also, the calculation of the goal_position was not right.
    
    Provides the relative-to-the-agent angular direction (rad) (0,2pi) to the goal with the observations as modeled by (y2 - y2 )/( x2 - x1)
    
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset()

        return self.observation(obs), info

    def observation(self, obs):
        # It is the absolute direction since the player does not have a direction
        # goal is...
        # -pi   -- behind
        # -pi/2 -- on the right
        # 0     -- in front
        # pi/2  -- on the left
        return self.observation_param(obs,self.env.goal_position,self.env.agent_pos)

    @staticmethod
    def observation_param(obs, goal_position, agent_pos):
        diff_row = abs(goal_position[0] - agent_pos[0])
        diff_col = abs(goal_position[1] - agent_pos[1])
        angle = np.arctan2(diff_row, diff_col)
        if goal_position[1] > agent_pos[1]:
            if goal_position[0] > agent_pos[0]: # (-pi, -pi/2)
                obs["goal_direction"] = -np.pi + angle
            else: # [pi/2,pi)
                obs["goal_direction"] = np.pi - angle
        else:
            if goal_position[0] > agent_pos[0]: # [-pi/2,0)
                obs["goal_direction"] = -np.pi/2 + (np.pi/2 - angle)
            else: # [0,pi/2)
                obs["goal_direction"] = angle
        return obs

class MinigridObservation(ObservationWrapper):
    """
    It only considers a square around the agent. The size must be odd.
    If the size is 3, the square looks like this:

    o o o
    o a o
    o o o

    """
    def __init__(self, env, size, view_size = 3):
        super().__init__(env)

        if view_size%2 ==0:
            raise Exception(f"view_size {view_size} must be odd.")
        
        self.view_size = view_size
        self.size = size

        flat_size = view_size*view_size + 1
        lows = np.zeros((flat_size,))
        highs = np.ones((flat_size,))
        highs = highs * 5 # largest index of tile type

        self.observation_space = Box(
            low=lows,
            high=highs,
            shape=(flat_size,),
            dtype="float32",
        )
    def observation(self, obs: ObsType, **kwargs):
        goal_direction = obs[-1]
        try:
            col, row = self.env.unwrapped.agent_pos
            col -= 1
            row -= 1
        except:
            found = False
            for row in range(self.size):
                for col in range(self.size):
                    if obs[col+row*self.size] == PLAYER:
                        found = True
                        break
                if found:
                    break
            # if it is not found, agent is on goal
            if not found:
                for row in range(self.size):
                    for col in range(self.size):
                        if obs[col+row*self.size] == TARGET:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    print(f"Player not found {obs}")
                    assert(False)

        return self.create_obs_square(goal_direction,obs,self.view_size, row, col,self.size)

    @staticmethod
    def create_obs_square(goal_direction,obs,view_size, row, col,size):
        obs_square = np.zeros((view_size, view_size))
        view_size_half = int((view_size - 1) / 2)
        obs_idxs = range(-view_size_half, view_size_half + 1)
        for i in obs_idxs:
            for j in obs_idxs:
                row_obs = row + i
                col_obs = col + j
                row_obs_square = i + view_size_half
                col_obs_square = j + view_size_half
                if row_obs < 0 or col_obs < 0 or row_obs > size - 1 or col_obs > size - 1:
                    obs_square[row_obs_square][col_obs_square] = WALL
                else:
                    obs_square[row_obs_square][col_obs_square] = obs[row_obs * size + col_obs]

        return np.concatenate((obs_square.flatten(), np.array((goal_direction,))))

class UnflattenObservationSquare(ObservationWrapper):
    """
    
    """
    def __init__(self, env, size, view_size = 3):
        super().__init__(env)

        if view_size%2 ==0:
            raise Exception(f"view_size {view_size} must be odd.")
        
        self.view_size = view_size

        lows = env.observation_space.low[:-1].reshape((1,view_size,view_size))
        highs = env.observation_space.high[:-1].reshape((1,view_size,view_size))

        image_observation_space = Box(
            low=lows,
            high=highs,
            shape=(1, view_size, view_size),
            dtype="float32",
        )
        goal_direction_space = Box(
            low=0,
            high=1,
            shape=(1,),
            dtype="float32",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "goal_direction": goal_direction_space,
            }
        )

    def observation(self, obs: ObsType, **kwargs):
        return dict(
            {
                "image": self.unflatten_obs_square(obs, self.view_size),
                "goal_direction": np.array([obs[-1]])
            }
        )
    
    @staticmethod
    def unflatten_obs_square(obs, view_size):
        obs_array = np.zeros((1, view_size, view_size))
        for i in range(view_size):
            for j in range(view_size):
                obs_array[0][i][j] = obs[i*view_size + j]
        return obs_array

def create_gardener_env(size, pctg_walls, pctg_phenomena, pctg_frogs, view_size, instance_seed=None, positions=None, steps_per_start_position=None):
    # steps = n_positions_reaching_goal * steps_per_start_position
    env = GardenerEnv(instSize=size, instance_seed=instance_seed,
                      max_steps=size*8, pctg_walls=pctg_walls, pctg_phenomena=pctg_phenomena, 
                      pctg_frogs=pctg_frogs, steps_per_start_position=steps_per_start_position)
    
    # if instance_seed is not None:
    #     image_shape = env.observation_space.spaces["image"].shape
    #     env = SimpleStateWrapper(env,image_shape[0], image_shape[1])
    #     assert image_shape[0] == image_shape[1]
    #     env = OheWrapper(env,image_shape[0], image_shape[1],image_shape[0]+1,values_start_at_one=False)
    # else:
    env = DirectionAngleObsWrapper(env)
    env = FlattenObservation(env)
    env = MinigridObservation(env, size, view_size)
    # env = UnflattenObservationSquare(env, size, view_size)
    return env
