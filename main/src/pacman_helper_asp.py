
import clingo
from clingo import Function, Number

from sb3_ext.clingoHelper import ClingoHelper
from oftendeeprl.sb3_ext.pacman_helper import PacmanClingoHelper


#TODO rename
class PacmanClingoHelperAsp(PacmanClingoHelper):

    def __init__(self, horizon, radius, ghosts, vegetarian, num_violations):
        self.ctl = clingo.Control(
            ["-c", f"horizon={horizon}",
             "-c", f"radius={radius}",
             "-c", f"num_violations={num_violations}",
             "-c", f"ghosts={ghosts}"])
        self.ctl.load('pacman_program_asp.lp')
        self.ctl.ground([("base", [])], context=self)

        self.radius = radius
        self.next = None
        self.penalty = 0
        self.vegetarian = vegetarian


    def set_clingo_externals(self, state):
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

            

    def less_violations_possible(self, state):
        self.next = None

        # Reset the clingo window
        self.reset_clingo_externals(state)

        # Set all externals of the currently considered window (see section
        # "Optimization-> Windowing" in the main document for more information)
        # Externals include cell information (walls, ghosts) as well as
        # information about policy preferences
        self.set_clingo_externals(state)

        # solve the LP
        print(self.ctl.solve(on_model=self.on_model))

    
        return None