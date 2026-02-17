# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"
import math
import queue
from functools import cache, lru_cache

from collections import defaultdict

import gymnasium.spaces
import numpy as np
from PIL.ImageChops import offset

from gym_pacman_rules.envs.game import Directions, Actions
import gym_pacman_rules.envs.util
from gym_pacman_rules.envs.pacman import PacmanRules, SCARED_TIME, COLLISION_TOLERANCE
from gym_pacman_rules.envs.util import Counter,nearestPoint


def features_dict_to_array(features : Counter, ordering = None):
    pac_map = features.pop('map',None)
    sorted_features = sorted(list(features.items()),key=lambda x: x[0])
    ext_features = list(filter(lambda x : x[0].startswith("#-of-ghosts-1-step-away-") or
                      x[0].startswith("#-of-scared-ghosts   -1-step-away-"), sorted_features))
    non_ext_features = list(filter(lambda x : not(x[0].startswith("#-of-ghosts-1-step-away-") or
                      x[0].startswith("#-of-scared-ghosts-1-step-away-")), sorted_features))
    #sorted_features = non_ext_features + ext_features
    other_features = np.array(list(zip(*sorted_features))[1])
    return other_features
    if pac_map is not None:
        return np.concatenate((other_features,pac_map.flatten()))
    else:
        return other_features

class FeatureExtractor:
    def __init__(self, height,width):
        self.width = width
        self.height = height

    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        raise Exception("Not implemented")
    def get_obs_space(self,nr_ghosts):
        raise Exception("Not implemented")

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestCapsule(pos, capsules, walls,legal_neighbor_cache = None):
    if len(capsules) == 0:
        return -1,-1
    fringe = [(pos[0], pos[1], 0, None)] # additionally return direction of first move toward food
    expanded = set()
    while fringe:
        pos_x, pos_y, dist,dir = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if (pos_x,pos_y) in capsules:
            return dist,dir
        # otherwise spread out from the location to its neighbours
        if legal_neighbor_cache is not None:
            nbrs = legal_neighbor_cache.get((pos_x, pos_y),None)
            if nbrs is None:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                legal_neighbor_cache[(pos_x,pos_y)] = nbrs
            else:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            next_dir = dir
            if next_dir is None:
                if nbr_y - pos_y == 1:
                    next_dir = Directions.NORTH
                if nbr_y - pos_y == -1:
                    next_dir = Directions.SOUTH
                if nbr_x - pos_x == 1:
                    next_dir = Directions.EAST
                if nbr_x - pos_x == -1:
                    next_dir = Directions.WEST
            fringe.append((nbr_x, nbr_y, dist+1,next_dir))


def closestFood(pos, food, walls, legal_neighbor_cache = None, return_dir=False):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0, None)] # additionally return direction of first move toward food
    expanded = set()
    while fringe:
        pos_x, pos_y, dist,dir = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            if return_dir:
                return dist,dir
            else:
                return dist
        # otherwise spread out from the location to its neighbours
        if legal_neighbor_cache is not None:
            nbrs = legal_neighbor_cache.get((pos_x, pos_y),None)
            if nbrs is None:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                legal_neighbor_cache[(pos_x,pos_y)] = nbrs
            else:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            next_dir = dir
            if next_dir is None:
                if nbr_y - pos_y == 1:
                    next_dir = Directions.NORTH
                if nbr_y - pos_y == -1:
                    next_dir = Directions.SOUTH
                if nbr_x - pos_x == 1:
                    next_dir = Directions.EAST
                if nbr_x - pos_x == -1:
                    next_dir = Directions.WEST
            fringe.append((nbr_x, nbr_y, dist+1,next_dir))
    # no food found
    return None

def ghostDistance(pac, ghost, walls, legal_neighbor_cache = None, return_dir = False):

    # fringe = [(pac[0], pac[1], 0, None)]
    fringe = queue.SimpleQueue()
    fringe.put((pac[0], pac[1], 0, None))
    expanded = set()
    while not fringe.empty():
        pos_x, pos_y, dist,dir = fringe.get() #fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if abs(pos_x - ghost[0]) <= COLLISION_TOLERANCE and abs(pos_y - ghost[1]) <= COLLISION_TOLERANCE:
            offset = abs(pos_x - ghost[0]) + abs(pos_y - ghost[1])
            if return_dir:
                return dist + offset,dir
            else:
                return dist + offset
        # otherwise spread out from the location to its neighbours
        if legal_neighbor_cache is not None:
            nbrs = legal_neighbor_cache.get((pos_x, pos_y),None)
            if nbrs is None:
                nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
                legal_neighbor_cache[(pos_x,pos_y)] = nbrs
        else:
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)

        for nbr_x, nbr_y in nbrs:
            next_dir = dir
            if next_dir is None:
                if nbr_y - pos_y == 1:
                    next_dir = Directions.NORTH
                if nbr_y - pos_y == -1:
                    next_dir = Directions.SOUTH
                if nbr_x - pos_x == 1:
                    next_dir = Directions.EAST
                if nbr_x - pos_x == -1:
                    next_dir = Directions.WEST
            # fringe.append((nbr_x, nbr_y, dist+1,next_dir))
            fringe.put((nbr_x, nbr_y, dist+1,next_dir))
    print(f"PAC: {pac}")
    print(f"ghost: {ghost}")

    raise Exception("no ghost found")

class HungryExtractor(FeatureExtractor):

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of non-scared ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and not g.isScared()) for g in ghosts)

        # count the number of scared ghosts 1-step away
        features["#-of-scared-ghosts-1-step-away"] = sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and g.isScared()) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features



class BlueExtractor(FeatureExtractor):

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of non-scared ghosts 1-step away
        features["#-of-ghosts1-1-step-away"] = ((next_x, next_y) in Actions.getLegalNeighbors(ghosts[0].getPosition(), walls) and not ghosts[0].isScared())

        # count the number of scared ghosts 1-step away
        features["#-of-scared-ghosts1-1-step-away"] = ((next_x, next_y) in Actions.getLegalNeighbors(ghosts[0].getPosition(), walls) and ghosts[0].isScared())

        # count the number of non-scared ghosts 1-step away
        features["#-of-ghosts2-1-step-away"] = (
                    (next_x, next_y) in Actions.getLegalNeighbors(ghosts[1].getPosition(), walls) and not ghosts[
                1].isScared())

        # count the number of scared ghosts 1-step away
        features["#-of-scared-ghosts2-1-step-away"] = (
                    (next_x, next_y) in Actions.getLegalNeighbors(ghosts[1].getPosition(), walls) and ghosts[
                1].isScared())

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

class ExtendedExtractor6(FeatureExtractor):
    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        if action is None: # action is None upon reset
            next_x, next_y = x,y
        else:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)

        # compute possible next locations for pacman
        neighbors = Actions.getLegalNeighbors((next_x, next_y), walls)
        s_n2 = 0
        s_sc2 = 0
        for n in neighbors:
            s_n2 += sum((n in Actions.getLegalNeighbors(g.getPosition(), walls) and not g.isScared()) for g in ghosts)
            s_sc2 += sum((n in Actions.getLegalNeighbors(g.getPosition(), walls) and g.isScared()) for g in ghosts)
        s_n3 = 0
        s_sc3 = 0
        for n in neighbors:
            ns = Actions.getLegalNeighbors(n, walls)
            for i in ns:
                s_n3 += sum((i in Actions.getLegalNeighbors(g.getPosition(), walls) and not g.isScared()) for g in ghosts)
                s_sc3 += sum((i in Actions.getLegalNeighbors(g.getPosition(), walls) and g.isScared()) for g in ghosts)


        # count the number of non-scared ghosts 2-step away
        features["#-of-ghosts-2-steps-away"] = s_n2 + sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and not g.isScared()) for g in ghosts)

        # count the number of scared ghosts 2-step away
        features["#-of-scared-ghosts-2-steps-away"] = s_sc2 + sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and g.isScared()) for g in ghosts)

        # count the number of non-scared ghosts 2-step away
        features["#-of-ghosts-3-steps-away"] = s_n3 + s_n2 + sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and not g.isScared()) for g in ghosts)

        # count the number of scared ghosts 2-step away
        features["#-of-scared-ghosts-3-steps-away"] = s_sc3 + s_sc2 + sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and g.isScared()) for g in ghosts)

        # count the number of non-scared ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and not g.isScared()) for g in ghosts)

        # count the number of scared ghosts 1-step away
        features["#-of-scared-ghosts-1-step-away"] = sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and g.isScared()) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-steps-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        else:
            features["eats-food"] = 0.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        else:
            features["closest-food"] = features.max()

        features.divideAll(features.max()) # changed to argmax
        features["map"] = constract_map_array(self.height,self.width,state)
        return features

    def get_obs_space(self,nr_ghosts):
        other_obs_size = 10
        map_size = self.width * self.height
        obs_size = other_obs_size + map_size
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        high[other_obs_size:] = 3 + 4 + 4
        return gymnasium.spaces.Box(low=low,high=high)





class ExtendedExtractor7(FeatureExtractor):
    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()


        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        features["x"] = x
        features["y"] = y

        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0
                    #possible.append(dir)

        features["#-of-ghosts-1-step-away"] = 0
        features["#-of-scared-ghosts-1-step-away"] = 0
        for i,g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            g_x,g_y = g.getPosition()
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-x"] = g_x
            features[f"ghost-{i}-y"] = g_y
            ghost_vector_x = x-g_x
            ghost_vector_y = y-g_y
            ghost_dist = math.sqrt(ghost_vector_x**2 + ghost_vector_y**2)

            if ghost_dist < 1e-6:
                features[f"ghost-{i}-dist"] = 0
                features[f"ghost-{i}-dir"] = 0
            else:
                if ghost_vector_x > 0:
                    ghost_dir = math.atan(ghost_vector_y / ghost_vector_x)
                if ghost_vector_x < 0 and ghost_vector_y >= 0:
                    ghost_dir = math.atan(ghost_vector_y / ghost_vector_x) + math.pi
                if ghost_vector_x < 0 and ghost_vector_y < 0:
                    ghost_dir = math.atan(ghost_vector_y / ghost_vector_x) - math.pi
                if ghost_vector_x == 0 and ghost_vector_y > 0:
                    ghost_dir = math.pi / 2
                if ghost_vector_x == 0 and ghost_vector_y < 0:
                    ghost_dir = -math.pi / 2

                features[f"ghost-{i}-dist"] = ghost_dist
                features[f"ghost-{i}-dir"] = ghost_dir

            if abs(g_x - x) + abs(g_y - y) <= 1:
                if is_scared:
                    features["#-of-scared-ghosts-1-step-away"] += 1
                else:
                    features["#-of-ghosts-1-step-away"] += 1

        cap_dist,cap_dir = closestCapsule((x, y),state.getCapsules(),walls)
        features["closest-capsule-dist"] = cap_dist
        features["closest-capsule-dir"] = cap_dir

        dist_dir = closestFood((x, y), food, walls, return_dir=True)
        if dist_dir is not None:
            dist,dir = dist_dir
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) #/ (walls.width * walls.height)
            features["closest-food-dir"] = dir
        else:
            features["closest-food"] = features.max()
            features["closest-food-dir"] = 0

        # features["map"] = constract_map_array(self.height,self.width,state)
        return features

    def get_obs_space(self,nr_ghosts):
        other_obs_size = 13
        # map_size = self.width * self.height
        obs_size = other_obs_size + nr_ghosts * 5
        low = np.zeros(obs_size)
        high = np.ones(obs_size) * max(self.width,self.height)
        # high[other_obs_size:] = 3 + 4 + 4
        return gymnasium.spaces.Box(low=low,high=high)



class ExtendedExtractor8(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()

    def getFeatures(self, state, action, eaten_ghost=False):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = Counter()

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        features["x"] = x
        features["y"] = y

        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            if dir == Directions.STOP:
                continue
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0
                    #possible.append(dir)

        features["#-of-ghosts-1-step-away"] = 0
        features["#-of-scared-ghosts-1-step-away"] = 0
        for i,g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            g_x,g_y = g.getPosition()
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            features[f"ghost-{i}-x"] = g_x
            features[f"ghost-{i}-y"] = g_y
            g_dist, g_dir = ghostDistance((x,y),g.getPosition(),walls,legal_neighbor_cache = self.legal_neighbor_cache,return_dir=True)

            features[f"ghost-{i}-dist"] = g_dist
            # features[f"ghost-{i}-dir"] = g_dir if g_dir is not None else -1

            add_direction_ohe(features,g_dir if g_dir is not None else Directions.STOP,f"ghost-{i}-dir")
            add_direction_ohe(features,g.getDirection(),f"ghost-{i}-heading")
            # features[f"ghost-{i}-heading"] = g.getDirection()
            ghost_vector_x = g_x - x
            ghost_vector_y = g_y - y
            if abs(ghost_vector_x) <= COLLISION_TOLERANCE and abs(ghost_vector_y) <= COLLISION_TOLERANCE:
                ghost_approx_angle = 0
            else:
                if ghost_vector_x > 0:
                    ghost_angle = math.atan(ghost_vector_y / ghost_vector_x)
                elif ghost_vector_x < 0 and ghost_vector_y >= 0:
                    ghost_angle = math.atan(ghost_vector_y / ghost_vector_x) + math.pi
                elif ghost_vector_x < 0 and ghost_vector_y < 0:
                    ghost_angle = math.atan(ghost_vector_y / ghost_vector_x) - math.pi
                elif ghost_vector_x == 0 and ghost_vector_y > 0:
                    ghost_angle = math.pi / 2
                elif ghost_vector_x == 0 and ghost_vector_y < 0:
                    ghost_angle = -math.pi / 2

                # assign approximate angles in cardinal directions
                if math.pi / 2 - math.pi/8 <= ghost_angle <= math.pi / 2 + math.pi/8:
                    ghost_approx_angle = 1 # NORTH ( different directions than used for actions)
                elif math.pi / 4 - math.pi / 8 <= ghost_angle <= math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 2  # NORTHEAST
                elif - math.pi / 8 <= ghost_angle <= + math.pi / 8:
                    ghost_approx_angle = 3  # EAST
                elif -math.pi / 4 - math.pi / 8 <= ghost_angle <= -math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 4 # SOUTHEAST
                elif -math.pi / 2 - math.pi / 8 <= ghost_angle <= -math.pi / 2 + math.pi / 8:
                    ghost_approx_angle = 5  # SOUTH
                elif -3*math.pi / 4 - math.pi / 8 <= ghost_angle <= -3*math.pi / 4 + math.pi / 8:
                    ghost_approx_angle = 6  # SOUTHWEST
                elif ghost_angle <= -math.pi + math.pi / 8 or ghost_angle >= 3*math.pi / 4 + math.pi/8:
                    ghost_approx_angle = 7  # WEST
                elif 3*math.pi / 4 - math.pi/8 <= ghost_angle <= 3*math.pi / 4 + math.pi/8:
                    ghost_approx_angle = 8 # NORTHWEST
            add_angle_direction_ohe(features,ghost_approx_angle,f"ghost-{i}-angle")
                # features[f"ghost-{i}-angle"] = ghost_angle
        if abs(g_x - x) + abs(g_y - y) <= 1:
                if is_scared:
                    features["#-of-scared-ghosts-1-step-away"] += 1
                else:
                    features["#-of-ghosts-1-step-away"] += 1

        cap_dist,cap_dir = closestCapsule((x, y),state.getCapsules(),walls,legal_neighbor_cache = self.legal_neighbor_cache)
        features["closest-capsule-dist"] = cap_dist
        # features["closest-capsule-dir"] = cap_dir

        add_direction_ohe(features, cap_dir, "closest-capsule-dir")

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache = self.legal_neighbor_cache,return_dir=True)
        if dist_dir is not None:
            dist,dir = dist_dir
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = dist
            add_direction_ohe(features,dir,"closest-food-dir")
        else:
            features["closest-food"] = max(self.width,self.height)
            # features["closest-food-dir"] = 6
            add_direction_ohe(features,Directions.STOP,"closest-food-dir")

        # features["map"] = constract_map_array(self.height,self.width,state)
        features["prev_ghost_eaten"] = eaten_ghost

        return features

    def get_obs_space(self,nr_ghosts):
        other_obs_size = 19
        # map_size = self.width * self.height
        obs_size = other_obs_size + nr_ghosts * 22
        low = np.zeros(obs_size)
        high = np.ones(obs_size) * max(self.width,self.height)
        # high[other_obs_size:] = 3 + 4 + 4
        return gymnasium.spaces.Box(low=low,high=high)

def add_direction_ohe(features, direction, feature_name,with_out_stop=True):
    for d in Actions._directions.keys():
        if with_out_stop and d == Directions.STOP:
            continue
        else:
            features[f"{feature_name}-{d}"] = d == direction

def add_angle_direction_ohe(features, angle, feature_name):
    for d in range(0,9):
        features[f"{feature_name}-{d}"] = d == angle


def constract_map_array(height,width,game_state):
    map = np.zeros((width,height))
    for x in range(width):
        for y in range(height):
            food, walls = game_state.data.food, game_state.data.layout.walls
            if food[x][y]:
                map[x][y] = 2
            if walls[x][y]:
                map[x][y] = 3

    for agentState in game_state.data.agentStates:
        if agentState == None: continue
        if agentState.configuration == None: continue
        x, y = [int(i) for i in nearestPoint(agentState.configuration.pos)]
        agent_dir = agentState.configuration.direction
        if agentState.isPacman:
            map[x][y] = 3 + agent_dir
        else:
            map[x][y] = 3 + 4 + (agent_dir)

    for x, y in game_state.data.capsules:
        map[x][y] = 1
    return map

class ExtendedExtractor9(ExtendedExtractor8):
    def getFeatures(self, state, _action):
        features = super().getFeatures(state,_action)
        x, y = state.getPacmanPosition()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            features[f"#-of-ghosts-1-step-away-{action}"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghosts)
            features[f"#-of-scared-ghosts-1-step-away-{action}"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghosts if g.isScared())
        return features
    def get_obs_space(self,nr_ghosts):
        other_obs_size = 21
        # map_size = self.width * self.height
        obs_size = other_obs_size + nr_ghosts * 24
        obs_size += 8
        low = np.zeros(obs_size)
        high = np.ones(obs_size) * max(self.width,self.height)
        return gymnasium.spaces.Box(low=low,high=high)

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

# from OFTEN-DeepRL
class DeepRLCompleteExtractor(FeatureExtractor):
    def __init__(self, height, width):
        super().__init__(height, width)
        self.legal_neighbor_cache = dict()
        self.feature_order =  ['closest-food', 'closest-food-dir-0', 'closest-food-dir-1', 'closest-food-dir-2', 'closest-food-dir-3', 'poss-dir-0', 'poss-dir-1', 'poss-dir-2', 'poss-dir-3', 'ghost-0-scared', 'ghost-0-scaredtime', 'ghost-0-dist', 'ghost-0-dir-0', 'ghost-0-dir-1', 'ghost-0-dir-2', 'ghost-0-dir-3', 'ghost-0-heading-0', 'ghost-0-heading-1', 'ghost-0-heading-2', 'ghost-0-heading-3', 'ghost-1-scared', 'ghost-1-scaredtime', 'ghost-1-dist', 'ghost-1-dir-0', 'ghost-1-dir-1', 'ghost-1-dir-2', 'ghost-1-dir-3', 'ghost-1-heading-0', 'ghost-1-heading-1', 'ghost-1-heading-2', 'ghost-1-heading-3', '#-of-non-scared-ghosts-1-step-away', '#-of-scared-ghosts-1-step-away', '#-of-non-scared-ghosts-le3-step-away', '#-of-scared-ghosts-le3-step-away', '#-of-non-scared-ghosts-1-step-away-0', '#-of-scared-ghosts-1-step-away-0', '#-of-non-scared-ghosts-le3-step-away-0', '#-of-scared-ghosts-le3-step-away-0', 'closest-food-0', '#-of-non-scared-ghosts-1-step-away-1', '#-of-scared-ghosts-1-step-away-1', '#-of-non-scared-ghosts-le3-step-away-1', '#-of-scared-ghosts-le3-step-away-1', 'closest-food-1', '#-of-non-scared-ghosts-1-step-away-2', '#-of-scared-ghosts-1-step-away-2', '#-of-non-scared-ghosts-le3-step-away-2', '#-of-scared-ghosts-le3-step-away-2', 'closest-food-2', '#-of-non-scared-ghosts-1-step-away-3', '#-of-scared-ghosts-1-step-away-3', '#-of-non-scared-ghosts-le3-step-away-3', '#-of-scared-ghosts-le3-step-away-3', 'closest-food-3', 'closest-capsule-dist', 'closest-capsule-dir-0', 'closest-capsule-dir-1', 'closest-capsule-dir-2', 'closest-capsule-dir-3', 'x', 'y', 'prev_ghost_eaten']


    def obs_from_state(self, state, action, eaten_ghost):
        features = self.getFeatures(state, action, eaten_ghost)
        return np.array([features[name] for name in self.feature_order])

    def getFeatures(self, state, action, eaten_ghost):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        n_ghosts = len(ghosts)
        features = Counter()
        max_dist = self.height + self.width

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()

        dist_dir = closestFood((x, y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache,
                           return_dir=True)
        if dist_dir is not None:
            dist,food_dir = dist_dir
            features[f"closest-food"] = dist / max_dist
            add_direction_ohe(features, food_dir, f"closest-food-dir",with_out_stop=True)
        else:
            features[f"closest-food"] = 1
            add_direction_ohe(features, Directions.STOP, f"closest-food-dir",with_out_stop=True)
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        for dir, vec in Actions._directionsAsList:
            if dir == Directions.STOP:
                continue
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if next_x < walls.width and next_y < walls.height:
                if not walls[next_x][next_y]:
                    features[f"poss-dir-{dir}"] = 1
                else:
                    features[f"poss-dir-{dir}"] = 0

        ghost_distances = defaultdict(list)
        for i,g in enumerate(ghosts):
            is_scared = 1 if g.isScared() else 0
            features[f"ghost-{i}-scared"] = is_scared
            features[f"ghost-{i}-scaredtime"] = g.scaredTimer / SCARED_TIME
            g_dist, g_dir = ghostDistance((x,y),g.getPosition(),walls,legal_neighbor_cache = self.legal_neighbor_cache,return_dir=True)
            ghost_distances["curr"].append((g_dist,g.isScared()))
            features[f"ghost-{i}-dist"] = g_dist / max_dist

            add_direction_ohe(features,g_dir if g_dir is not None else Directions.STOP,f"ghost-{i}-dir",with_out_stop=True)
            add_direction_ohe(features,g.getDirection(),f"ghost-{i}-heading",with_out_stop=True)
            for action in Actions._directions.keys():
                if action == Directions.STOP:
                    continue
                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(x + dx), int(y + dy)

                g_dist, g_dir = ghostDistance((next_x, next_y), g.getPosition(), walls,
                                              legal_neighbor_cache=self.legal_neighbor_cache, return_dir=True)
                ghost_distances[action].append((g_dist,g.isScared()))

        features["#-of-non-scared-ghosts-1-step-away"] = len([(d,sc) for (d,sc) in ghost_distances["curr"] if not sc and d <= 1])
        features["#-of-scared-ghosts-1-step-away"] = len([(d,sc) for (d,sc) in ghost_distances["curr"] if sc and d <= 1])
        features["#-of-non-scared-ghosts-le3-step-away"] = len([(d,sc) for (d,sc) in ghost_distances["curr"] if not sc and d <= 3])
        features["#-of-scared-ghosts-le3-step-away"] = len([(d,sc) for (d,sc) in ghost_distances["curr"] if sc and d <= 3])


        for action in Actions._directions.keys():
            if action == Directions.STOP:
                continue

            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            features[f"#-of-non-scared-ghosts-1-step-away-{action}"] = (len(
                [(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 1]) -features["#-of-non-scared-ghosts-1-step-away"])/n_ghosts
            features[f"#-of-scared-ghosts-1-step-away-{action}"] = (len(
                [(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 1]) -features["#-of-scared-ghosts-1-step-away"])/n_ghosts
            features[f"#-of-non-scared-ghosts-le3-step-away-{action}"] = (len(
                [(d, sc) for (d, sc) in ghost_distances[action] if not sc and d <= 3]) -features["#-of-non-scared-ghosts-le3-step-away"])/n_ghosts
            features[f"#-of-scared-ghosts-le3-step-away-{action}"] = (len(
                [(d, sc) for (d, sc) in ghost_distances[action] if sc and d <= 3]) - features["#-of-scared-ghosts-le3-step-away"])/n_ghosts

            if action == Directions.STOP:
                continue
            dist = closestFood((next_x, next_y), food, walls, legal_neighbor_cache=self.legal_neighbor_cache,
                               return_dir=False)
            if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features[f"closest-food-{action}"] = (dist - features[f"closest-food"]) / max_dist
            else:
                features[f"closest-food-{action}"] = 1 # self.height + self.width


        features["#-of-non-scared-ghosts-1-step-away"] /=n_ghosts
        features["#-of-scared-ghosts-1-step-away"] /=n_ghosts
        features["#-of-non-scared-ghosts-le3-step-away"] /=n_ghosts
        features["#-of-scared-ghosts-le3-step-away"]/=n_ghosts
        cap_dist,cap_dir = closestCapsule((x, y),state.getCapsules(),walls,legal_neighbor_cache = self.legal_neighbor_cache)
        features["closest-capsule-dist"] = cap_dist

        add_direction_ohe(features, cap_dir, "closest-capsule-dir",with_out_stop=True)

        features["x"] = x / self.width
        features["y"] = y / self.height

        features["prev_ghost_eaten"] = eaten_ghost

        return features

    def get_obs_space(self,nr_ghosts):
        other_obs_size = 41
        obs_size = other_obs_size + nr_ghosts * 11
        low = np.zeros(obs_size)
        high = np.ones(obs_size)
        return gymnasium.spaces.Box(low=low,high=high)