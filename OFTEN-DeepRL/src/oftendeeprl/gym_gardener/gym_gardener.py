import copy
import random

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium import spaces

from typing import Any, SupportsFloat
import numpy as np

from .instance import Instance, create_random_instance

EMPTY = 0
FROG = 1
PLANT = 2
WALL = 3
PLAYER = 4
TARGET = 5

# ---------------
# (1,1) (1,2) ...
# (2,1) (2,2) ...
#  ...   ...  ... 
# ---------------

class GardenerEnv(gym.Env):
    """
    
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        instSize: int,
        instance_seed: int = None,
        max_steps: int = 10_000,
        pctg_walls: float = 0.25,
        pctg_phenomena: float = 0.05,
        pctg_frogs: float = 0.5,
        randomize = False,
        reward_goal = 100,
        reward_moving = -1,
        reward_not_moving = -10,
        reward_plant_killed = 5,
        reward_frog_killed = 5,
        steps_per_start_position = None,
        random_starts = False
    ):
        
        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        self.step_count_one_start_position = 0

        self.reward_goal = reward_goal
        self.reward_moving = reward_moving
        self.reward_not_moving = reward_not_moving
        self.reward_plant_killed = reward_plant_killed
        self.reward_frog_killed = reward_frog_killed

        self.number_dead_frogs = 0
        self.number_dead_plants = 0

        self.pctg_walls = pctg_walls
        self.pctg_phenomena = pctg_phenomena
        self.pctg_frogs = pctg_frogs

        # Environment configuration
        self.instSize = instSize

        # deterministic or nondeterministic
        self.setting = "deterministic" if pctg_frogs <= 0 else "nondeterministic"

        # Action enumeration for this environment
        self.actions_player = [0, 1, 2, 3]
        self.actions_frog = [0, 1, 2, 3]

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions_player))

        # Observations are dictionaries containing an
        # encoding of the grid
        image_observation_space = spaces.Box(
            low=0,
            high=5, # empty, wall, plant, frog, goal, player
            shape=(self.instSize, self.instSize),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
            }
        )

        # Current position of the agent
        self.agent_pos: np.ndarray | tuple[int, int] = None

        # Goal position
        self.goal_position: np.ndarray | tuple[int, int] = None
        self.randomize = randomize

        number_instance = 0 # this was originally a counter
        prefix = ""
        if instance_seed is not None:
            random.seed(instance_seed)
        self.instance = create_random_instance(self.instSize, self.setting, number_instance,
                                               self.pctg_walls, self.pctg_phenomena,
                                               self.pctg_frogs, prefix)
        self.frog_copy = copy.deepcopy(self.instance.frogs)
        self.start_position = self.instance.player
        if not self.instance.target_reachable():
            raise Exception("Target unreachable, change seed/configuration.")

        self.random_starts = random_starts
        if steps_per_start_position is not None or random_starts:
            self.positions_reaching_goal = self.create_starting_states(self.instance)
            random.shuffle(self.positions_reaching_goal)
        else:
            self.positions_reaching_goal = None
        self.steps_per_start_position = steps_per_start_position

        if self.positions_reaching_goal is not None:
            self.positions_reaching_goal.remove(self.instance.player)


    def create_starting_states(self, instance):
        change = True
        reachable = [instance.target]
        actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        while change:
            change = False
            for c in reachable:
                for a in actions:
                    candidate = (c[0] + a[0], c[1] + a[1])
                    if (0 < candidate[0] <= instance.size and 0 < candidate[
                        1] <= instance.size and candidate not in reachable and
                            candidate not in instance.walls):
                        reachable.append(candidate)
                        change = True
        reachable.remove(instance.target)
        return reachable
    
    def feasible_actions(self):
        actions = []
        if self.instance.is_feasible((self.instance.player[0], self.instance.player[1] + 1)):
            actions.append(0) # go south
        if self.instance.is_feasible((self.instance.player[0], self.instance.player[1] - 1)):
            actions.append(1) # go north
        if self.instance.is_feasible((self.instance.player[0] - 1, self.instance.player[1])):
            actions.append(2) # go west
        if self.instance.is_feasible((self.instance.player[0] + 1, self.instance.player[1])):
            actions.append(3) # go east

        return actions

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        This is similar to gardener's next step but with just the basic funcionality
        """
    
        self.step_count += 1
        self.step_count_one_start_position += 1
        terminated = False
        truncated = False
        reached_goal = False
        info = dict()
        
        previous_position_player = self.agent_pos
        previous_number_dead_frogs = self.number_dead_frogs
        previous_number_dead_plants = self.number_dead_plants
        self.number_dead_frogs
        if action in self.feasible_actions():
            self.instance.execute(action)
            self.agent_pos = self.instance.player
        else:
            pass # remain in the same position
        self.instance.emulate_frogs()
        self.instance.check_violations()

        self.number_dead_frogs = len(self.instance.dead_frogs)
        self.number_dead_plants = len(self.instance.dead_plants)

        frog_killed = True if self.number_dead_frogs > previous_number_dead_frogs else False
        plant_killed = True if self.number_dead_plants > previous_number_dead_plants else False

        if self.goal_position == self.agent_pos:
            # if the player reaches the goal
            terminated = True
            reached_goal = True
            reward = self.reward_goal
        elif frog_killed:
            reward = self.reward_frog_killed
        elif plant_killed:
            reward = self.reward_plant_killed
        elif previous_position_player != self.agent_pos:
            # if the player moves
            reward = self.reward_moving
        else:
            # if the player does not move, i.e. goes to a wall
            reward = self.reward_not_moving

        if self.step_count >= self.max_steps:
            truncated = True

        observation = self.gen_obs()
        info["reached_goal"] = reached_goal
        info["nr-killed-frogs"] = self.number_dead_frogs
        info["nr-killed-plants"] = self.number_dead_plants

        return observation, reward, terminated, truncated, info

    def get_state(self):
        return self.instance

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """
        
        """
        self.number_dead_frogs = 0
        self.number_dead_plants = 0

        number_instance = 0 # this was originally a counter
        prefix = ""
        if self.randomize:
            while True:
                self.instance = create_random_instance(self.instSize, self.setting, number_instance,
                                                          self.pctgWalls, self.pctgPhenomena,
                                                          self.pctgFrogs, prefix)
                if self.instance.target_reachable():
                    break

            self.agent_pos = self.instance.player
        else:
            if self.random_starts:
                self.start_position = random.choice(self.positions_reaching_goal)
            else:
                self.instance.frogs = copy.deepcopy(self.frog_copy)
                if self.positions_reaching_goal is not None and self.step_count_one_start_position >= self.steps_per_start_position:
                    if len(self.positions_reaching_goal)==0:
                        raise Exception(f"Wrong configuration of total steps (learning) and steps per start position.")
                    self.start_position = self.positions_reaching_goal.pop(0)
                    self.step_count_one_start_position = 0
            self.agent_pos = self.start_position
            self.instance.player = self.start_position
            self.instance.dead_frogs = []
            self.instance.dead_plants = []

        self.goal_position = self.instance.target

        super().reset()

        obs = self.gen_obs()
        # Step count since episode start
        self.step_count = 0
        self.instance.visited = {self.instance.player: 1}

        return obs, {}

    def gen_obs(self):
        return self.gen_obs_param(self.instance)

    @staticmethod
    def gen_obs_param(instance):
        # convert from instance to array
        instSize = instance.size
        agent_pos = instance.player
        goal_position = instance.target
        obs = [[EMPTY] * instSize for i in range(instSize)]

        for i,frog in enumerate(instance.frogs):
            if i not in instance.dead_frogs:
                obs[frog[1] - 1][frog[0] - 1] = FROG
        for plant in instance.plants:
            if plant not in instance.dead_plants:
                obs[plant[1] - 1][plant[0] - 1] = PLANT
        for wall in instance.walls:
            obs[wall[1] - 1][wall[0] - 1] = WALL

        obs[agent_pos[1] - 1][agent_pos[0] - 1] = PLAYER
        obs[goal_position[1] - 1][goal_position[0] - 1] = TARGET

        # obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}
        return {"image": obs}

    def render(self): # -> RenderFrame | list[RenderFrame] | None: 
        """
        This should call gardener's interface
        """
        from interface import Interface
        self.interface = Interface(instance=self.instance)
        
        raise NotImplementedError

    def close(self):
        """
        
        """
        # pass
        raise NotImplementedError
