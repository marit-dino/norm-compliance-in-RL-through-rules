import clingo
import numpy as np
from clingo import Number, Function

from gym_gardener.gym_gardener import GardenerEnv, FROG, EMPTY
from gym_gardener.utils import get_current_time_ms
from gym_gardener.wrappers import DirectionAngleObsWrapper, FlattenObservation, MinigridObservation, \
    UnflattenObservationSquare
from sb3_ext.LearningInferface import LearningInferface
from sb3_ext.clingoHelper import ClingoHelper


class GardenerHelper(ClingoHelper):
    def __init__(self, horizon, radius, only_frogs = False):
        # instance = instance
        self.radius = radius
        self.horizon = horizon
        self.time_diff = []
        self.ctl = None
        self.cache = []
        self.only_frogs = only_frogs

    def on_model(self, m):
        self.save_solution(m)
        show = " ".join([str(i) for i in m.symbols(shown=True)])
        #print("Answer:\n{}".format(show))

    def save_solution(self, m):
        self.player = []
        self.final = False
        for sym in m.symbols(shown=True):
            match sym.name:
                case "action_taken":
                    if len(sym.arguments) == 4:
                        while len(self.player) < sym.arguments[1].number + 1:
                            self.player.append(0)
                        self.player[sym.arguments[1].number] = sym.arguments[
                            0].number

    def setup(self,instance):
        self.max_frogs = min(pow((self.radius * 2 + 1), 2) - 1,
                             len(instance.frogs))
        self.ctl = clingo.Control(["-c", f"horizon={self.horizon}", "-c",
                                   f"frogs={self.max_frogs}", "-c",
                                   f"radius={self.radius}", "-c",
                                   f"size={instance.size}"])
        self.ctl.load('gardener_program.lp')
        self.ctl.ground([("base", [])], context=self)

    def reset_clingo_externals(self,instance):
        for i in range(1, 11):
            self.ctl.assign_external(Function("multi", [
                Number(i)
            ]), False)
        for i in range(1, instance.size + 1):
            for j in range(1, instance.size + 1):
                self.ctl.assign_external(Function("player", [
                    Number(i),
                    Number(j)
                ]), False)
        for c, f in enumerate(instance.frogs):
            self.ctl.assign_external(Function("foutside", [
                Number(c)
            ]), False)
        for i in range(self.radius + 1):
            for c, f in enumerate(instance.frogs):
                self.ctl.assign_external(Function("fcol", [
                    Number(c),
                    Number(i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("fcol", [
                    Number(c),
                    Number(-i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("frow", [
                    Number(c),
                    Number(i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("frow", [
                    Number(c),
                    Number(-i),
                    Number(0)
                ]), False)
            for j in range(self.radius + 1):
                self.ctl.assign_external(Function("wall", [
                    Number(i),
                    Number(j)
                ]), False)
                self.ctl.assign_external(Function("wall", [
                    Number(-i),
                    Number(j)
                ]), False)
                self.ctl.assign_external(Function("wall", [
                    Number(i),
                    Number(-j)
                ]), False)
                self.ctl.assign_external(Function("wall", [
                    Number(-i),
                    Number(-j)
                ]), False)
            if not self.only_frogs:
                for j in range(self.radius + 1):
                    self.ctl.assign_external(Function("plant", [
                        Number(i),
                        Number(j)
                    ]), False)
                    self.ctl.assign_external(Function("plant", [
                        Number(-i),
                        Number(j)
                    ]), False)
                    self.ctl.assign_external(Function("plant", [
                        Number(i),
                        Number(-j)
                    ]), False)
                    self.ctl.assign_external(Function("plant", [
                        Number(-i),
                        Number(-j)
                    ]), False)
            for j in range(self.radius + 1):
                self.ctl.assign_external(Function("target", [
                    Number(i),
                    Number(j)
                ]), False)
                self.ctl.assign_external(Function("target", [
                    Number(-i),
                    Number(j)
                ]), False)
                self.ctl.assign_external(Function("target", [
                    Number(i),
                    Number(-j)
                ]), False)
                self.ctl.assign_external(Function("target", [
                    Number(-i),
                    Number(-j)
                ]), False)
            for j in range(self.radius + 1):
                for a in range(4):
                    for r in range(4):
                        self.ctl.assign_external(Function("action", [
                            Number(a),
                            Number(r),
                            Number(i),
                            Number(j)
                        ]), False)
                        self.ctl.assign_external(Function("action", [
                            Number(a),
                            Number(r),
                            Number(-i),
                            Number(j)
                        ]), False)
                        self.ctl.assign_external(Function("action", [
                            Number(a),
                            Number(r),
                            Number(i),
                            Number(-j)
                        ]), False)
                        self.ctl.assign_external(Function("action", [
                            Number(a),
                            Number(r),
                            Number(-i),
                            Number(-j)
                        ]), False)

    def get_relevant_states(self, instance,_obs_param):
        neighboring = []
        orig_player_pos = instance.player
        midpoint = instance.player
        for i in range(-self.radius,self.radius + 1):
            for j in range(-self.radius,self.radius + 1):
                absolut = (midpoint[0] + i, midpoint[1] + j)
                instance.player = absolut
                if 1 <= absolut[0] <= instance.size and 1 <= absolut[1] <= instance.size and absolut not in instance.walls:
                    obs = GardenerEnv.gen_obs_param(instance)
                    obs = DirectionAngleObsWrapper.observation_param(obs,instance.target,instance.player)
                    obs = FlattenObservation.observation_param(obs, normalize_direction=True)
                    if i != 0 and j != 0:
                        for k in range(obs.shape[0]-1):
                            if obs[k] == FROG:
                                obs[k] = EMPTY
                    obs = MinigridObservation.create_obs_square(goal_direction=obs[-1],obs=obs,view_size=9, row=absolut[0],
                                                                col=absolut[1],size=instance.size)
                    neighboring.append((obs,absolut))
                else:
                    neighboring.append((None,absolut))
        instance.player = orig_player_pos
        return neighboring
    def set_clingo_externals(self,instance,action_ranks):
        midpoint = instance.player
        # player
        self.ctl.assign_external(
            Function("player", [Number(midpoint[0]),
                                Number(midpoint[1])]), True)

        multi = instance.visited[midpoint] if midpoint in instance.visited else 0
        self.ctl.assign_external(Function("multi", [
            Number(multi)
        ]), True)

        # walls & plants
        for i in range(self.radius + 1):
            for j in range(self.radius + 1):
                absolut = (midpoint[0] + i, midpoint[1] + j)
                for a in range(4):
                    r = action_ranks[((absolut[0], absolut[1]),a)]
                    # self.learning.get_action_rank((absolut[0], absolut[1]),
                    #                                   a)
                    self.ctl.assign_external(Function("action", [
                        Number(a),
                        Number(r),
                        Number(i),
                        Number(j)
                    ]), True)
                if absolut == instance.target:
                    self.ctl.assign_external(
                        Function("target", [Number(i), Number(j)]), True)
                if absolut in instance.plants and absolut not in instance.dead_plants:
                    self.ctl.assign_external(
                        Function("plant", [Number(i), Number(j)]), True)
                if absolut in instance.walls:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(j)]), True)
                if absolut[0] < 1 or absolut[1] < 1 or absolut[
                    0] > instance.size or absolut[
                    1] > instance.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(j)]), True)

                absolut = (midpoint[0] - i, midpoint[1] + j)
                for a in range(4):
                    r = action_ranks[((absolut[0], absolut[1]),a)]
                    # self.learning.get_action_rank((absolut[0], absolut[1]),
                    #                                   a)
                    self.ctl.assign_external(Function("action", [
                        Number(a),
                        Number(r),
                        Number(-i),
                        Number(j)
                    ]), True)
                if absolut == instance.target:
                    self.ctl.assign_external(
                        Function("target", [Number(-i), Number(j)]), True)
                if absolut in instance.plants and absolut not in instance.dead_plants:
                    self.ctl.assign_external(
                        Function("plant", [Number(-i), Number(j)]), True)
                if absolut in instance.walls:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(j)]), True)
                if absolut[0] < 1 or absolut[1] < 1 or absolut[
                    0] > instance.size or absolut[
                    1] > instance.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(j)]), True)

                absolut = (midpoint[0] + i, midpoint[1] - j)
                for a in range(4):
                    r = action_ranks[((absolut[0], absolut[1]),a)]
                    # self.learning.get_action_rank((absolut[0], absolut[1]),
                    #                                   a)
                    self.ctl.assign_external(Function("action", [
                        Number(a),
                        Number(r),
                        Number(i),
                        Number(-j)
                    ]), True)
                if absolut == instance.target:
                    self.ctl.assign_external(
                        Function("target", [Number(i), Number(-j)]), True)
                if absolut in instance.plants and absolut not in instance.dead_plants:
                    self.ctl.assign_external(
                        Function("plant", [Number(i), Number(-j)]), True)
                if absolut in instance.walls:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(-j)]), True)
                if absolut[0] < 1 or absolut[1] < 1 or absolut[
                    0] > instance.size or absolut[
                    1] > instance.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(-j)]), True)

                absolut = (midpoint[0] - i, midpoint[1] - j)
                for a in range(4):
                    r = action_ranks[((absolut[0], absolut[1]),a)]
                    # self.learning.get_action_rank((absolut[0], absolut[1]),
                    #                                   a)
                    self.ctl.assign_external(Function("action", [
                        Number(a),
                        Number(r),
                        Number(-i),
                        Number(-j)
                    ]), True)
                if absolut == instance.target:
                    self.ctl.assign_external(
                        Function("target", [Number(-i), Number(-j)]), True)
                if absolut in instance.plants and absolut not in instance.dead_plants:
                    self.ctl.assign_external(
                        Function("plant", [Number(-i), Number(-j)]), True)
                if absolut in instance.walls:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(-j)]), True)
                if absolut[0] < 1 or absolut[1] < 1 or absolut[
                    0] > instance.size or absolut[
                    1] > instance.size:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(-j)]), True)

        # frogs
        count_frog = 0
        for c, f in enumerate(instance.frogs):
            relative = (f[0] - midpoint[0], f[1] - midpoint[1])
            if abs(relative[0]) > self.radius or abs(
                    relative[
                        1]) > self.radius or c in instance.dead_frogs:
                #    self.ctl.assign_external(Function("foutside", [Number(c)]),
                #                             True)
                pass
            else:
                self.ctl.assign_external(
                    Function("fcol",
                             [Number(count_frog), Number(relative[0]),
                              Number(0)]),
                    True)
                self.ctl.assign_external(
                    Function("frow",
                             [Number(count_frog), Number(relative[1]),
                              Number(0)]),
                    True)
                count_frog += 1
        for i in range(count_frog, self.max_frogs):
            self.ctl.assign_external(Function("foutside", [Number(i)]),
                                     True)
    def value_pairs_to_ranks(self,actionValuePairs):
        action_ranks = dict()
        for state in actionValuePairs.keys():
            action_values = actionValuePairs[state]
            sorted_values = sorted(action_values,key=lambda av : av[1])
            for rank,(action,value) in enumerate(sorted_values):
                action_ranks[(state,action)] = rank
        return action_ranks


    def get_action(self, state,actionValuePairs):

        time_pre_compute = get_current_time_ms()
        instance = state
        action_ranks = self.value_pairs_to_ranks(actionValuePairs)
        fall_back_action = None
        for a in range(4):
            if action_ranks[(instance.player,a)] == 3:
                fall_back_action  = a
        assert fall_back_action is not None
        # Failsafe: If the framework is stuck in a loop, follow the RL
        if instance.player in instance.visited and instance.visited[instance.player] > 10:
            # self.cache.clear()
            return fall_back_action # self.learning.get_action(instance)

        # If there are any cached actions use them fist
        # if len(self.cache) > 0:
        #     action = self.cache.pop(0)
        #     time_post_compute = get_current_time_ms()
        #     time_diff = time_post_compute - time_pre_compute
        #     self.time_diff.append(time_diff)
        #     return action

        self.player = []

        # Reset the clingo window
        self.reset_clingo_externals(instance)

        # Set all externals of the currently considered window (see section
        # "Optimization-> Windowing" in the main document for more information)
        # Externals include cell information (walls, frogs, plants) as well as
        # information about policy preferences
        self.set_clingo_externals(instance,action_ranks)

        # solve the LP
        self.ctl.solve(on_model=self.on_model)

        # measure the computation time
        time_post_compute = get_current_time_ms()
        time_diff = time_post_compute - time_pre_compute
        self.time_diff.append(time_diff)

        # cache further actions if desired and return current next action
        action = self.player[0]
        # if cache > 1:
        #     for i in range(1, cache):
        #         rel = self.player[i]
        #         self.cache.append(rel)
        return action

    # def get_reward(self, a, c, r):
    #     rank = self.learning.get_action_rank((c.number, r.number), a.number)
    #     return Number(rank)
