
import clingo
from clingo import Function, Number

from sb3_ext.clingoHelper import ClingoHelper


class PacmanClingoHelper(ClingoHelper):

    def __init__(self, horizon, radius, ghosts, norm):
        self.ctl = clingo.Control(
            ["-c", f"horizon={horizon}",
             "-c", f"radius={radius}",
             "-c", f"ghosts={ghosts}"])
        self.ctl.load('../../OFTEN-DeepRL/src/oftendeeprl/pacman_program.lp')
        self.ctl.ground([("base", [])], context=self)
        
        self.horizon = horizon
        self.radius = radius
        self.next = None
        self.penalty = 0
        self.vegetarian = "vegetarian" == norm
        self.permissive = "permissive" == norm
        self.vegan = "vegan" == norm

    def get_relevant_states(self, state,obs):
        return [(obs,"curr")]

    def on_model(self, m):
        self.save_solution(m)
        self.show = " ".join([str(i) for i in m.symbols(shown=True)])
        #print("Answer:\n{}".format(self.show))

    def save_solution(self, m):
        self.penalty = 0
        self.next = [0,0]
        for sym in m.symbols(shown=True):
            if sym.name == "pcol":
                if sym.arguments[1].number == 1:
                    self.next[0] = sym.arguments[0].number
            if sym.name == "prow":
                if sym.arguments[1].number == 1:
                    self.next[1] = sym.arguments[0].number
            if sym.name == "penalty":
                self.penalty = sym.arguments[0].number
        # if self.next[0] != -1 and self.next[1] != -1:
        #    self.next = (next[0], next[1])

    def reset_clingo_externals(self, state):
        # reset outside ghosts
        for c, g in enumerate(state.getGhostPositions()):
            self.ctl.assign_external(Function("goutside", [
                Number(c)
            ]), False)
        # reset actions
        for a in range(4):
            for r in range(4):
                self.ctl.assign_external(Function("action", [
                    Number(a),
                    Number(r)
                ]), False)
        for i in range(self.radius + 1):
            # reset inside ghosts
            for c, g in enumerate(state.getGhostPositions()):
                self.ctl.assign_external(Function("gcol", [
                    Number(c),
                    Number(i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("gcol", [
                    Number(c),
                    Number(-i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("grow", [
                    Number(c),
                    Number(i),
                    Number(0)
                ]), False)
                self.ctl.assign_external(Function("grow", [
                    Number(c),
                    Number(-i),
                    Number(0)
                ]), False)
            # reset walls
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
        
        for t in range (0, self.horizon):
            self.ctl.assign_external(Function("eaten_ghost", [Number(t)]), False)


    def set_clingo_externals(self, state, actionValuePairs, eaten_ghost_before):
        midpoint = state.getPacmanPosition()

        # walls
        for i in range(self.radius + 1):
            for j in range(self.radius + 1):
                absolut = (midpoint[0] + i, midpoint[1] + j)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= state.getWalls().width or absolut[
                    1] >= state.getWalls().height:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(j)]), True)
                elif state.getWalls()[absolut[0]][absolut[1]]:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(j)]), True)
                absolut = (midpoint[0] - i, midpoint[1] + j)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= state.getWalls().width or absolut[
                    1] >= state.getWalls().height:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(j)]), True)
                elif state.getWalls()[absolut[0]][absolut[1]]:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(j)]), True)
                absolut = (midpoint[0] + i, midpoint[1] - j)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= state.getWalls().width or absolut[
                    1] >= state.getWalls().height:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(-j)]), True)
                elif state.getWalls()[absolut[0]][absolut[1]]:
                    self.ctl.assign_external(
                        Function("wall", [Number(i), Number(-j)]), True)
                absolut = (midpoint[0] - i, midpoint[1] - j)
                if absolut[0] < 0 or absolut[1] < 0 or absolut[
                    0] >= state.getWalls().width or absolut[
                    1] >= state.getWalls().height:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(-j)]), True)
                elif state.getWalls()[absolut[0]][absolut[1]]:
                    self.ctl.assign_external(
                        Function("wall", [Number(-i), Number(-j)]), True)

        # ghosts
        for c, g in enumerate(state.getGhostPositions()):
            relative = (g[0] - midpoint[0], g[1] - midpoint[1])
            if abs(relative[0]) > self.radius or abs(
                    relative[1]) > self.radius:
                self.ctl.assign_external(Function("goutside", [Number(c)]),
                                         True)
            elif self.vegetarian and c == 0:
                self.ctl.assign_external(Function("goutside", [Number(c)]),
                                         True)
            else:
                self.ctl.assign_external(
                    Function("gcol",
                             [Number(c), Number(int(relative[0])), Number(0)]),
                    True)
                self.ctl.assign_external(
                    Function("grow",
                             [Number(c), Number(int(relative[1])), Number(0)]),
                    True)

        # encode the policy preferences
        sortedActionValuePairs = sorted(actionValuePairs, key=lambda x: x[1],
                                        reverse=True)

        for a in range(4):
            name = None
            if a == 0:
                name = 'East'
            if a == 1:
                name = 'West'
            if a == 2:
                name = 'North'
            if a == 3:
                name = 'South'
            r = 3
            for i, actionValuePair in enumerate(sortedActionValuePairs):
                if actionValuePair[0] == name:
                    r = i
            self.ctl.assign_external(Function("action",
                                              [Number(a),
                                               Number(r)]), True)

        if self.permissive and eaten_ghost_before:
            self.ctl.assign_external(Function("eaten_ghost", [Number(0)]), True)


    def get_action(self, state, actionValuePairs, eaten_ghost_before):
        self.next = None
        actionValuePairs = actionValuePairs["curr"]

        # Reset the clingo window
        self.reset_clingo_externals(state)

        # Set all externals of the currently considered window (see section
        # "Optimization-> Windowing" in the main document for more information)
        # Externals include cell information (walls, ghosts) as well as
        # information about policy preferences
        self.set_clingo_externals(state, actionValuePairs, eaten_ghost_before)

        # solve the LP
        self.ctl.solve(on_model=self.on_model)

        # return the next action
        if 0 < self.next[0]:
            return 'East'
        if 0 > self.next[0]:
            return 'West'
        if 0 < self.next[1]:
            return 'North'
        if 0 > self.next[1]:
            return 'South'
        return None