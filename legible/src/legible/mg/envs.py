import random
from abc import ABC
from enum import Enum, IntEnum

from gymnasium.spaces import Discrete
from minigrid.core.constants import TILE_PIXELS, OBJECT_TO_IDX, COLOR_TO_IDX, COLORS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj, Goal, Lava, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_line

from minigrid.core.world_object import OBJECT_TO_IDX
OBJECT_TO_IDX["slippery"] = 11


class MiniGridEnvWithSlipperyAndLava(MiniGridEnv, ABC):
    def __init__(
            self,
            mission_space: MissionSpace,
            grid_size: int = None,
            width: int = None,
            height: int = None,
            max_steps: int = 100,
            see_through_walls: bool = False,
            agent_view_size: int = 7,
            render_mode: str = None,
            screen_size: int = 640,
            highlight: bool = True,
            tile_size: int = TILE_PIXELS,
            agent_pov: bool = False,
    ):
        super().__init__(mission_space,
                         grid_size,
                         width,
                         height,
                         max_steps,
                         see_through_walls,
                         agent_view_size,
                         render_mode,
                         screen_size,
                         highlight,
                         tile_size,
                         agent_pov)

    def step(self, action):
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move forward
        if action == self.actions.forward and fwd_cell is not None and fwd_cell.type == "slippery":
            slip_prob = fwd_cell.slip_prob
            if random.random() < slip_prob:
                slip_dir = fwd_cell.slip_dir
                # print("slipped")
                x = fwd_pos[0]
                y = fwd_pos[1]
                neighbors = self.get_neighbors(slip_dir,x,y)
                reachable_neigh = [n for n in neighbors if self.grid.get(*n) is None or self.grid.get(*n).can_overlap()]
                if len(reachable_neigh) == 0:
                    # this can happen in the wall and lava environment with balls, so we slipped into an "enemy"
                    obs, reward, terminated, truncated, info = super().step(self.actions.done)
                    return obs, -1, True, truncated,info
                self.agent_pos = tuple(random.choice(reachable_neigh))
                # print(f"Entered {self.agent_pos} instead of {fwd_pos}")
                obs, reward, terminated, truncated, info = super().step(self.actions.done)
                if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == "goal":
                    terminated = True
                    reward = self._reward()
                if self.grid.get(*self.agent_pos) is not None and self.grid.get(*self.agent_pos).type == "lava":
                    terminated = True
                return obs, reward, terminated, truncated, info
            else:
                return super().step(action)
        else:
            return super().step(action)

    def get_main_neighbors(self, x,y):
        return [(x, y - 1),(x, y + 1),(x+1, y ),(x-1,y)]

    def get_neighbors(self, slip_dir,x,y):
        if slip_dir == SlipDir.NORTH:
            return [(x,y-1)]
        elif slip_dir == SlipDir.SOUTH:
            return [(x, y + 1)]
        elif slip_dir == SlipDir.EAST:
            return [(x+1,y)]
        elif slip_dir == SlipDir.WEST:
            return [(x-1,y)]
        elif slip_dir == SlipDir.NORTHEAST:
            return [(x,y-1), (x+1,y), (x+1,y-1)]  # have third tile as well
        elif slip_dir == SlipDir.NORTHWEST:
            return [(x,y-1), (x-1,y), (x-1,y-1)]
        elif slip_dir == SlipDir.SOUTHEAST:
            return [(x,y+1),(x+1,y),(x+1,y+1)]
        elif slip_dir == SlipDir.SOUTHWEST:
            return [(x,y+1),(x-1,y),(x-1,y+1)]

class SlipDir(IntEnum):
    NORTH = 1
    NORTHEAST = 2
    EAST = 3
    SOUTHEAST = 4
    SOUTH = 5
    SOUTHWEST = 6
    WEST = 7
    NORTHWEST = 8


class Slippery(WorldObj):

    def __init__(self, slip_dir : SlipDir, slip_prob: float):
        super().__init__("slippery", "blue")
        self.slip_dir = slip_dir
        self.slip_prob = slip_prob

    def can_overlap(self):
        return True

    def encode(self):
        return OBJECT_TO_IDX[self.type],COLOR_TO_IDX[self.color],int(self.slip_dir)

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
        arrow_col = (255, 0, 0) if self.slip_prob > 0.3 else (255,165,0)

        if self.slip_dir == SlipDir.NORTH:
            fill_coords(img, point_in_line(0.5, 0.2, 0.5, 0.8, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.4, 0.3, 0.5, 0.2, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.6, 0.3, 0.5, 0.2, r=0.075), arrow_col)
        if self.slip_dir == SlipDir.SOUTHEAST:
            fill_coords(img, point_in_line(0.2, 0.2, 0.8, 0.8, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.7, 0.8, 0.8, 0.8, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.8, 0.7, 0.8, 0.8, r=0.075), arrow_col)
        if self.slip_dir == SlipDir.NORTHWEST:
            fill_coords(img, point_in_line(0.2, 0.2, 0.8, 0.8, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.3, 0.2, 0.2, 0.2, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.2, 0.3, 0.2, 0.2, r=0.075), arrow_col)
        if self.slip_dir == SlipDir.NORTHEAST:
            fill_coords(img, point_in_line(0.2, 0.8, 0.8, 0.2, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.7, 0.2, 0.8, 0.2, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.8, 0.3, 0.8, 0.2, r=0.075), arrow_col)
        if self.slip_dir == SlipDir.SOUTHWEST:
            fill_coords(img, point_in_line(0.2, 0.8, 0.8, 0.2, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.3, 0.8, 0.2, 0.8, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.2, 0.7, 0.2, 0.8, r=0.075), arrow_col)
        if self.slip_dir == SlipDir.WEST:
            fill_coords(img, point_in_line(0.2, 0.5, 0.8, 0.5, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.3, 0.4, 0.2, 0.5, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.3, 0.6, 0.2, 0.5, r=0.075), arrow_col)
        if self.slip_dir == SlipDir.EAST:
            fill_coords(img, point_in_line(0.2, 0.5, 0.8, 0.5, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.7, 0.4, 0.8, 0.5, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.7, 0.6, 0.8, 0.5, r=0.075), arrow_col)
        if self.slip_dir == SlipDir.SOUTH:
            fill_coords(img, point_in_line(0.5, 0.2, 0.5, 0.8, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.4, 0.7, 0.5, 0.8, r=0.075), arrow_col)
            fill_coords(img, point_in_line(0.6, 0.7, 0.5, 0.8, r=0.075), arrow_col)

class ZigZagEnv(MiniGridEnvWithSlipperyAndLava):
    def __init__(
        self,
        zig_zags,
        agent_start_pos=(1, 3),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.zig_zags = zig_zags
        self.width = zig_zags * 6 + 2 + 2 # 6 tiles is one zig zag + two columns for start and finish and two columns for walls
        self.height = 6

        if max_steps is None:
            max_steps = 4 * self.width * self.height

        super().__init__(
            mission_space=mission_space,
            grid_size=None,
            width = self.width,
            height=self.height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.action_space = Discrete(self.actions.forward + 1)

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        offset = 1
        for i in range(self.zig_zags):
            # lower lava
            self.put_obj(Lava(), offset + 2, 4)
            self.put_obj(Lava(), offset + 3, 4)
            self.put_obj(Slippery(SlipDir.SOUTHWEST, 0.4), offset + 2, 3)
            self.put_obj(Slippery(SlipDir.SOUTHWEST, 0.4), offset + 3, 3)

            path_slip_directions = (SlipDir.WEST,SlipDir.EAST) if random.randint(0,1) == 0 else (SlipDir.EAST,SlipDir.WEST)
            self.put_obj(Slippery(path_slip_directions[0], 0.8), offset + 2, 2)
            self.put_obj(Slippery(path_slip_directions[0], 0.8), offset + 3, 2)
            self.put_obj(Slippery(path_slip_directions[1], 0.8), offset + 2, 1)
            self.put_obj(Slippery(path_slip_directions[1], 0.8), offset + 3, 1)

            # upper lava
            self.put_obj(Lava(), offset + 5, 1)
            self.put_obj(Lava(), offset + 6, 1)
            self.put_obj(Slippery(SlipDir.NORTHWEST,0.4),offset+5,2)
            self.put_obj(Slippery(SlipDir.NORTHWEST,0.4),offset+6,2)

            path_slip_directions = (SlipDir.WEST,SlipDir.EAST) if random.randint(0,1) == 0 else (SlipDir.EAST,SlipDir.WEST)
            self.put_obj(Slippery(path_slip_directions[0], 0.8), offset + 5, 3)
            self.put_obj(Slippery(path_slip_directions[0], 0.8), offset + 6, 3)
            self.put_obj(Slippery(path_slip_directions[1], 0.8), offset + 5, 4)
            self.put_obj(Slippery(path_slip_directions[1], 0.8), offset + 6, 4)

            offset += 6

        self.mission = "get to the green goal square"


def north_south_slippery(slip_prob):
    return Slippery(SlipDir.NORTH if random.randint(0,1) == 1 else SlipDir.SOUTH, slip_prob)


class WallsAndLava(MiniGridEnvWithSlipperyAndLava):
    def __init__(
            self,
            rows=1, # size 1 or 2
            columns=1,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            max_steps: int | None = None,
            randomize_danger=True,
            **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.rows = rows
        self.columns = columns
        self.width = 4 + columns * 4 + 4 + 2
        self.height = 3 + rows * 4 + 3 + 2
        self.randomize_danger = randomize_danger
        if max_steps is None:
            max_steps = 8 * self.width * self.height

        super().__init__(
            mission_space=mission_space,
            grid_size=None,
            width=self.width,
            height=self.height,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.obstacles = []
        self.action_space = Discrete(self.actions.forward + 1)

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        ################
        #           24d#
        #           222#
        # #### ### ###2#
        #      2d2  222#
        #222 2d2       #
        #2## ### ##### #
        #242           #
        #dd2           #
        ################

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, 2)

        self.obstacles = []
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # upper part should always look like this
        ###############
        #          24l#
        #          222# where 2 is slight slippery north west, 4 is stronger,
        # middle part should always be like that
        #### ### ###2#
        #    2d2  222#
        #    222     #
        #222         #
        #2## ### #####
        # lower part should always be like that
        #242        G#
        #dd2         #
        ##############
        # columns affect the middle part

        # upper part
        self.put_obj(Slippery(SlipDir.NORTHEAST,0.25),4 + 4 * self.columns + 2,1)
        self.put_obj(Slippery(SlipDir.EAST,0.4),4 + 4 * self.columns + 3,1)
        self.put_obj(Lava(),4 + 4 * self.columns + 4, 1)

        self.put_obj(Slippery(SlipDir.NORTHEAST,0.25),4 + 4 * self.columns + 2,2)
        self.put_obj(Slippery(SlipDir.NORTHEAST,0.25),4 + 4 * self.columns + 3,2)
        self.put_obj(Slippery(SlipDir.NORTH,0.4),4 + 4 * self.columns + 4, 2)
        # middle
        j_offset = 3
        danger_right = True if not self.randomize_danger else random.randint(0,1) == 0
        for r in range(self.rows):
            # self.put_obj(Wall(), 2, j_offset)

            self.put_obj(north_south_slippery(0.2), 1, j_offset)
            self.put_obj(north_south_slippery(0.2), 2, j_offset)
            self.put_obj(Wall(), 3, j_offset)
            self.put_obj(Wall(), 4, j_offset)
            self.put_obj(Wall(), 3, j_offset+4)
            self.put_obj(Wall(), 4, j_offset+4)
            c_offset = 5
            for c in range(self.columns):
                # self.put_obj(Wall(), c_offset, j_offset)
                self.put_obj(north_south_slippery(0.2), c_offset, j_offset)
                self.put_obj(Wall(), c_offset + 1, j_offset)
                self.put_obj(Wall(), c_offset + 2, j_offset)
                self.put_obj(north_south_slippery(0.2), c_offset+3, j_offset)

                c_offset += 4

            c_offset = 5
            danger_up = True if not self.randomize_danger else random.randint(0,1) == 0
            for c in range(self.columns):
                if danger_up:
                    self.put_obj(Lava(), c_offset+1, j_offset+1)
                    self.put_obj(Slippery(SlipDir.EAST,0.4), c_offset, j_offset+1)
                    self.put_obj(Slippery(SlipDir.WEST,0.4), c_offset+2, j_offset+1)

                    self.put_obj(Slippery(SlipDir.NORTHEAST,0.25), c_offset, j_offset+2)
                    self.put_obj(Slippery(SlipDir.NORTH,0.4), c_offset+1, j_offset+2)
                    self.put_obj(Slippery(SlipDir.NORTHWEST,0.25), c_offset+2, j_offset+2)
                else:
                    self.put_obj(Lava(), c_offset + 1, j_offset + 3)
                    self.put_obj(Slippery(SlipDir.EAST, 0.4), c_offset, j_offset + 3)
                    self.put_obj(Slippery(SlipDir.WEST, 0.4), c_offset + 2, j_offset + 3)

                    self.put_obj(Slippery(SlipDir.SOUTHEAST, 0.25), c_offset, j_offset + 2)
                    self.put_obj(Slippery(SlipDir.SOUTH, 0.4), c_offset + 1, j_offset + 2)
                    self.put_obj(Slippery(SlipDir.SOUTHWEST, 0.25), c_offset + 2, j_offset + 2)
                danger_up = not danger_up if not self.randomize_danger else random.randint(0,1) == 0
                c_offset += 4

            # self.put_obj(Wall(), c_offset, j_offset)
            self.put_obj(north_south_slippery(0.2), c_offset, j_offset)
            self.put_obj(Wall(), c_offset +1, j_offset)
            self.put_obj(Wall(), c_offset +2, j_offset)

            # self.put_obj(Wall(), c_offset, j_offset+4)
            self.put_obj(north_south_slippery(0.2), c_offset, j_offset+4)
            self.put_obj(Wall(), c_offset + 1, j_offset+4)
            self.put_obj(Wall(), c_offset +2, j_offset+4)


            lava_spot_right = 4 + 4 * self.columns + 4
            if danger_right:
                self.put_obj(Lava(), lava_spot_right, j_offset+2)
                self.put_obj(Slippery(SlipDir.SOUTH,0.4), 1, j_offset + 2)
            else:
                self.put_obj(Lava(), 1, j_offset + 2)
                self.put_obj(Slippery(SlipDir.SOUTH,0.4), lava_spot_right, j_offset+2)

            self.put_obj(Slippery(SlipDir.SOUTHWEST,0.4), lava_spot_right, j_offset+1)
            self.put_obj(Slippery(SlipDir.NORTHWEST,0.4), lava_spot_right, j_offset+3)
            self.put_obj(Slippery(SlipDir.SOUTHEAST,0.4), lava_spot_right-1, j_offset+2)

            self.put_obj(Slippery(SlipDir.SOUTHEAST,0.4), 1, j_offset+1)
            self.put_obj(Slippery(SlipDir.NORTHEAST,0.4), 1, j_offset+3)
            self.put_obj(Slippery(SlipDir.SOUTHWEST,0.4), 2, j_offset+2)
            danger_right = not danger_right if not self.randomize_danger else random.randint(0,1) == 0
            c_offset = 5
            for c in range(self.columns):
                # self.put_obj(Wall(), c_offset, j_offset+4)

                self.put_obj(north_south_slippery(0.2), c_offset, j_offset+4)
                self.put_obj(Wall(), c_offset + 1, j_offset +4)
                self.put_obj(Wall(), c_offset + 2, j_offset +4)
                self.put_obj(north_south_slippery(0.2), c_offset+3, j_offset+4)
                c_offset += 4
            obstacle = Ball("red")
            self.obstacles.append(obstacle)
            self.put_obj(obstacle,2,j_offset+1)
            j_offset += 4
        # lower part

        self.put_obj(Slippery(SlipDir.SOUTH,0.4), 1, j_offset+2)
        self.put_obj(Slippery(SlipDir.SOUTHWEST,0.25), 2, j_offset+2)
        self.put_obj(Slippery(SlipDir.SOUTHWEST,0.25), 3, j_offset+2)
        self.put_obj(Lava(), 1, j_offset + 3)
        self.put_obj(Slippery(SlipDir.WEST,0.4), 2, j_offset + 3)
        self.put_obj(Slippery(SlipDir.WEST,0.25), 3, j_offset + 3)
        self.put_obj(Goal(),4 + 4 * self.columns + 4,j_offset+3)

    def step(self, action):
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)
        obs, reward, terminated, truncated, info = super().step(action)
        # Move forward
        if action == self.actions.forward and fwd_cell is not None and fwd_cell.type == "ball":
            reward = -1
            terminated = True
            return obs,reward,terminated,truncated, info
        else:
            for o in self.obstacles:
                x,y = o.cur_pos
                neighbors = self.get_main_neighbors(x,y)
                empty_neighbors = [(x_n,y_n) for (x_n,y_n) in neighbors if (self.grid.get(x_n,y_n) is None or
                                   self.grid.get(x_n,y_n).type == "empty")
                                   and self.agent_pos[0] != x_n and self.agent_pos[1] != y_n]
                if len(empty_neighbors) == 0:
                    continue
                next_pos = random.choice(empty_neighbors)
                self.put_obj(o,next_pos[0],next_pos[1])
                self.grid.set(x,y, None)

            return obs,reward,terminated,truncated, info
def main():
    env = WallsAndLava(rows=1,columns=1,render_mode="human")
    # env = ZigZagEnv(zig_zags=4,render_mode="human")
    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
