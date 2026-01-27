import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import copy

headless = True

from .featureExtractors import ExtendedExtractor6, features_dict_to_array, ExtendedExtractor7, ExtendedExtractor8, \
    ExtendedExtractor9, DeepRLCompleteExtractor
# from scipy.special import kwargs

if not headless:
    from .graphicsDisplay import PacmanGraphics, DEFAULT_GRID_SIZE

from .game import Actions
from .pacman import ClassicGameRules
from .layout import getLayout, getRandomLayout

from .ghostAgents import DirectionalGhost
from .pacmanAgents import OpenAIAgent

from gymnasium.utils import seeding

import json
import os

if not headless:
    from .rendering_support import SimpleImageViewer

DEFAULT_GHOST_TYPE = 'DirectionalGhost'

MAX_GHOSTS = 5

PACMAN_ACTIONS = list(range(5)) #['North', 'South', 'East', 'West', 'Stop']

PACMAN_DIRECTIONS = list(range(4)) #['North', 'South', 'East', 'West']
ROTATION_ANGLES = [0, 180, 90, 270]

MAX_EP_LENGTH = 400

import os
fdir = '/'.join(os.path.split(__file__)[:-1])
print(fdir)
layout_params = json.load(open(fdir + '/../../layout_params.json'))

print("Layout parameters")
print("------------------")
for k in layout_params:
    print(k,":",layout_params[k])
print("------------------")

class PacmanEnv(gymnasium.Env):
    layouts = [
        'capsuleClassic', 'contestClassic', 'mediumClassic', 'mediumGrid', 'minimaxClassic', 'openClassic',
        'originalClassic', 'smallClassic', 'capsuleClassic', 'smallGrid', 'testClassic', 'trappedClassic', 'trickyClassic'
    ]

    noGhost_layouts = [l + '_noGhosts' for l in layouts]

    MAX_MAZE_SIZE = (7, 7)
    num_envs = 1

    observation_space = spaces.Box(low=0, high=255,
            shape=(84, 84, 3), dtype=np.uint8)

    def __init__(self,**kwargs):
        self.action_space = spaces.Discrete(4) # up, down, left right
        if not headless:
            self.display = PacmanGraphics(1.0)
        else:
            self.display = None
        self._action_set = range(len(PACMAN_ACTIONS))
        self.location = None
        self.viewer = None
        # self.done = False
        self.terminated = False
        self.truncated = False
        self.layout = None
        if "layout" in kwargs:
            self.layout = kwargs["layout"]
        self.chooseLayout(randomLayout=False,chosenLayout=self.layout)

        self.max_ep_len = MAX_EP_LENGTH
        if "originalClassic" in kwargs["layout"]:
            print("Increase max ep len")
            self.max_ep_len *= 2
        if "features" in kwargs:
            self.features = kwargs["features"]
        else:
            self.features = "image"
            self.observation_space = spaces.Box(low=0, high=255,
                                           shape=(84, 84, 3), dtype=np.uint8)
        if self.features != "image":
            self.instantiate_feature_extractor(kwargs["layout"])
        self.np_random = None
        if headless:
            self.do_render = False
        else:
            self.do_render = "render_mode" in kwargs and kwargs["render_mode"] == "human"
        

    def setObservationSpace(self):
        screen_width, screen_height = self.display.calculate_screen_dimensions(self.layout.width,   self.layout.height)
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(int(screen_height),
                int(screen_width),
                3), dtype=np.uint8)

    def chooseLayout(self, randomLayout=False,
        chosenLayout=None, no_ghosts=True):

        if randomLayout:
            self.layout = getRandomLayout(layout_params, self.np_random)
        else:
            if chosenLayout is None:
                if not no_ghosts:
                    chosenLayout = self.np_random.choice(self.layouts)
                else:
                    chosenLayout = self.np_random.choice(self.noGhost_layouts)
            self.chosen_layout = chosenLayout
            # print("Chose layout", chosenLayout)
            self.layout = getLayout(chosenLayout) if type(chosenLayout) == str else chosenLayout
        self.maze_size = (self.layout.width, self.layout.height)

    # def seed(self, seed=None):
    #     if self.np_random is None:
    #         self.np_random, seed = seeding.np_random(seed)
    #     self.chooseLayout(randomLayout=False,chosenLayout=self.layout)
    #     return [seed]

    def reset(self, layout=None,**kwargs):
        # get new layout
        #if self.layout is None:
        #    self.chooseLayout(randomLayout=True)

        self.step_counter = 0
        self.cum_reward = 0

        self.terminated = False
        self.truncated = False

        # we don't want super powerful ghosts
        self.ghosts = [DirectionalGhost( i+1, prob_attack=0.2, prob_scaredFlee=0.2) for i in range(MAX_GHOSTS)]

        # this agent is just a placeholder for graphics to work
        self.pacman = OpenAIAgent()

        self.rules = ClassicGameRules(300)
        self.rules.quiet = False

        self.game = self.rules.newGame(self.layout, self.pacman, self.ghosts,
            self.display, True, False, do_render=self.do_render)

        self.game.init()

        if self.do_render:
            self.display.initialize(self.game.state.data)
            self.display.updateView()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]
        self.ghostInFrame = any([np.sum(np.abs(np.array(g) - np.array(self.location))) <= 2 for g in self.ghostLocations])

        self.location_history = [self.location]
        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history = [self.orientation]
        self.illegal_move_counter = 0

        self.cum_reward = 0

        self.initial_info = {
            # 'past_loc': [self.location_history[-1]],
            # 'curr_loc': [self.location_history[-1]],
            # 'past_orientation': [[self.orientation_history[-1]]],
            # 'curr_orientation': [[self.orientation_history[-1]]],
            # 'illegal_move_counter': [self.illegal_move_counter],
            # 'ghost_positions': [self.ghostLocations],
            # 'ghost_in_frame': [self.ghostInFrame],
            'step_counter': [[0]],
        }

        return self.features.obs_from_state(self.game.state, None),self.initial_info


    def step(self, action):
        truncated = False
        # implement code here to take an action
        if self.step_counter >= self.max_ep_len or self.terminated or self.truncated:
            self.step_counter += 1
            self.truncated = self.truncated or self.step_counter >= self.max_ep_len
            # return (np.zeros(self.observation_space.shape)
            return self.null_obs(), 0.0, self.terminated,self.truncated, {
                # 'past_loc': [self.location_history[-2]],
                # 'curr_loc': [self.location_history[-1]],
                # 'past_orientation': [[self.orientation_history[-2]]],
                # 'curr_orientation': [[self.orientation_history[-1]]],
                # 'illegal_move_counter': [self.illegal_move_counter],
                'step_counter': [[self.step_counter]],
                # 'ghost_positions': [self.ghostLocations],
                'r': [self.cum_reward],
                'l': [self.step_counter],
                # 'ghost_in_frame': [self.ghostInFrame],
                'episode': {
                    'r': self.cum_reward,
                    'l': self.step_counter
                }
            }

        pacman_action = PACMAN_ACTIONS[action]

        legal_actions = self.game.state.getLegalPacmanActions()
        illegal_action = False
        if pacman_action not in legal_actions:
            self.illegal_move_counter += 1
            illegal_action = True
            pacman_action = 'Stop' # Stop is always legal

        reward = self.game.step(pacman_action)
        self.cum_reward += reward
        # reward shaping for illegal actions
        if illegal_action:
            reward -= 10

        terminated = self.game.state.isWin() or self.game.state.isLose()

        self.location = self.game.state.data.agentStates[0].getPosition()
        self.location_history.append(self.location)
        self.ghostLocations = [a.getPosition() for a in self.game.state.data.agentStates[1:]]

        self.orientation = PACMAN_DIRECTIONS.index(self.game.state.data.agentStates[0].getDirection())
        self.orientation_history.append(self.orientation)

        extent = (self.location[0] - 1, self.location[1] - 1),(self.location[0] + 1, self.location[1] + 1),
        self.ghostInFrame = any([ g[0] >= extent[0][0] and g[1] >= extent[0][1] and g[0] <= extent[1][0] and g[1] <= extent[1][1]
            for g in self.ghostLocations])
        self.step_counter += 1
        info = {
            # 'past_loc': [self.location_history[-2]],
            # 'curr_loc': [self.location_history[-1]],
            # 'past_orientation': [[self.orientation_history[-2]]],
            # 'curr_orientation': [[self.orientation_history[-1]]],
            # 'illegal_move_counter': [self.illegal_move_counter],
            'step_counter': [[self.step_counter]],
            'episode': None,
            # 'ghost_positions': [self.ghostLocations],
            # 'ghost_in_frame': [self.ghostInFrame],
        }

        if self.step_counter >= self.max_ep_len:
            truncated = True
        self.terminated = terminated
        self.truncated = truncated
        if self.terminated or self.truncated: # only if done, send 'episode' info
            info['episode'] = [{
                'r': self.cum_reward,
                'l': self.step_counter
            }]
        if not self.game.state.data.eaten_ghost and True in self.game.state.data._eaten[1:]:
            self.game.state.data.eaten_ghost = True
        return self.features.obs_from_state(self.game.state, action), reward, terminated, truncated, info

    def get_action_meanings(self):
        return [PACMAN_ACTIONS[i] for i in self._action_set]

    def get_image(self):
        return self._get_image(crop=False)

    # just change the get image function
    def _get_image(self, crop=True):
        # get x, y
        image = self.display.image

        w, h = image.size
        DEFAULT_GRID_SIZE_X, DEFAULT_GRID_SIZE_Y = w / float(self.layout.width), h / float(self.layout.height)

        extent = [
            DEFAULT_GRID_SIZE_X *  (self.location[0] - 1),
            DEFAULT_GRID_SIZE_Y *  (self.layout.height - (self.location[1] + 2.2)),
            DEFAULT_GRID_SIZE_X *  (self.location[0] + 2),
            DEFAULT_GRID_SIZE_Y *  (self.layout.height - (self.location[1] - 1.2))]
        extent = tuple([int(e) for e in extent])
        self.image_sz = (84,84)
        if crop:
            image = image.crop(extent).resize(self.image_sz)
        return np.array(image)

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            # from gymnasium.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        # TODO: implement code here to do closing stuff
        if not headless:
            if self.viewer is not None:
                self.viewer.close()
            self.display.finish()

    def __del__(self):
        self.close()

    def null_obs(self):
        if self.features == "image":
            return np.zeros(self.observation_space.shape)
        else:
            return self.observation_space.low #raise Exception(f"Not immplemented features {self.features}")

    def obs(self,state,action):
        if self.features == "image":
            return self._get_image()
        else:
            feature_dict = self.features.getFeatures(state, action)
            feature_array = features_dict_to_array(feature_dict)
            return feature_array
    def instantiate_feature_extractor(self, layout_name):
        nr_ghosts = 0
        if "medium" in layout_name or "small" in layout_name:
            nr_ghosts = 2
        elif "originalClassic" in layout_name:
            nr_ghosts = 4
        if self.features == "extended-6":
            self.features = ExtendedExtractor6(height=self.layout.height,width=self.layout.width)
            self.observation_space = self.features.get_obs_space(nr_ghosts)
        elif self.features == "extended-7":
            self.features = ExtendedExtractor7(height=self.layout.height,width=self.layout.width)
            self.observation_space = self.features.get_obs_space(nr_ghosts)
        elif self.features == "extended-8":
            self.features = ExtendedExtractor8(height=self.layout.height,width=self.layout.width)
            self.observation_space = self.features.get_obs_space(nr_ghosts)
        elif self.features == "extended-9":
            self.features = ExtendedExtractor9(height=self.layout.height,width=self.layout.width)
            self.observation_space = self.features.get_obs_space(nr_ghosts)
        elif self.features == "complete" :
            self.features = DeepRLCompleteExtractor(height=self.layout.height, width=self.layout.width)
            self.observation_space = self.features.get_obs_space(nr_ghosts)
        else:
            raise Exception(f"Feature extractor '{self.features}' Not implemented")
        
    
    def save_state(self):
        return {
            "game_state": copy.deepcopy(self.game.state),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "np_random": copy.deepcopy(self.np_random),
            "step_counter": self.step_counter,
            "cum_reward": self.cum_reward,
            "location":  self.location,
            "ghostLocations": copy.deepcopy(self.ghostLocations),
            "ghostInFrame": self.ghostInFrame,
            "location_history": copy.deepcopy(self.location_history),
            "orientation": self.orientation,
            "orientation_history": copy.deepcopy(self.orientation_history),
            "illegal_move_counter": self.illegal_move_counter        
            }
    
    def load_state(self, state):
        self.game.state = copy.deepcopy(state["game_state"])
        self.step_counter = state["step_counter"]
        self.cum_reward = state["cum_reward"]
        self.terminated = state["terminated"]
        self.truncated = state["truncated"]
        self.np_random = copy.deepcopy(state["np_random"])
        self.illegal_move_counter = state["illegal_move_counter"]
        self.location = state["location"]
        self.ghostLocations = copy.deepcopy(state["ghostLocations"])
        self.ghostInFrame = state["ghostInFrame"]
        self.location_history = copy.deepcopy(state["location_history"])
        self.orientation = state["orientation"]
        self.orientation_history = copy.deepcopy(state["orientation_history"])
