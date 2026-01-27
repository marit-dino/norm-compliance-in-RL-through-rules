
import clingo
from clingo import Function, Number

from sb3_ext.clingoHelper import ClingoHelper
from oftendeeprl.sb3_ext.pacman_helper import PacmanClingoHelper
import logging

log = logging.getLogger(__name__)

class PacmanViolationClingoHelper(PacmanClingoHelper):

    def __init__(self, horizon, radius, ghosts, norms, num_norms):
        self.ctl = clingo.Control(
            ["-c", f"max_horizon={horizon}",
             "-c", f"radius={radius}",
             "-c", f"num_norms={num_norms}",
             "-c", f"ghosts={ghosts}"])
        self.ctl.load('pacman_program_asp.lp')
        self.ctl.ground([("base", [])], context=self)

        self.radius = radius
        self.next = None
        self.penalty = 0
        self.vegetarian = "vegetarian" in norms
        self.permissive = "permissive" in norms
        self.num_norms = num_norms
        self.horizon = horizon


    def reset_clingo_externals(self, state):
        for v in range(0, self.num_norms * self.horizon):
            self.ctl.assign_external(Function("num_violations", [Number(v)]), False)
        
        for h in range(0, self.horizon):
            self.ctl.assign_external(Function("dynamic_horizon", [Number(h)]), False)

        return super().reset_clingo_externals(state)

    def set_clingo_externals(self, state, num_violations, dynamic_horizon, eaten_ghost_before):
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
                        #log.info(f"wall({i},{j}).")
                    elif state.getWalls()[absolut[0]][absolut[1]]:
                        self.ctl.assign_external(
                            Function("wall", [Number(i), Number(j)]), True)
                        #log.info(f"wall({i},{j}).")
                    absolut = (midpoint[0] - i, midpoint[1] + j)
                    if absolut[0] < 0 or absolut[1] < 0 or absolut[
                        0] >= state.getWalls().width or absolut[
                        1] >= state.getWalls().height:
                        self.ctl.assign_external(
                            Function("wall", [Number(-i), Number(j)]), True)
                        #log.info(f"wall({-i},{j}).")
                    elif state.getWalls()[absolut[0]][absolut[1]]:
                        self.ctl.assign_external(
                            Function("wall", [Number(-i), Number(j)]), True)
                        #log.info(f"wall({-i},{j}).")
                    absolut = (midpoint[0] + i, midpoint[1] - j)
                    if absolut[0] < 0 or absolut[1] < 0 or absolut[
                        0] >= state.getWalls().width or absolut[
                        1] >= state.getWalls().height:
                        self.ctl.assign_external(
                            Function("wall", [Number(i), Number(-j)]), True)
                        #log.info(f"wall({i},{-j}).")
                    elif state.getWalls()[absolut[0]][absolut[1]]:
                        self.ctl.assign_external(
                            Function("wall", [Number(i), Number(-j)]), True)
                        #log.info(f"wall({i},{-j}).")
                    absolut = (midpoint[0] - i, midpoint[1] - j)
                    if absolut[0] < 0 or absolut[1] < 0 or absolut[
                        0] >= state.getWalls().width or absolut[
                        1] >= state.getWalls().height:
                        self.ctl.assign_external(
                            Function("wall", [Number(-i), Number(-j)]), True)
                        #log.info(f"wall({-i},{-j}).")
                    elif state.getWalls()[absolut[0]][absolut[1]]:
                        self.ctl.assign_external(
                            Function("wall", [Number(-i), Number(-j)]), True)
                        #log.info(f"wall({-i},{-j}).")

            # ghosts
            for c, g in enumerate(state.getGhostPositions()):
                relative = (g[0] - midpoint[0], g[1] - midpoint[1])
                if abs(relative[0]) > self.radius or abs(
                        relative[1]) > self.radius:
                    self.ctl.assign_external(Function("goutside", [Number(c)]),
                                            True)
                    #log.info(f"goutside({c}).")
                elif self.vegetarian and c == 0:
                    self.ctl.assign_external(Function("goutside", [Number(c)]),
                                            True)
                    #log.info(f"goutside({c}).")
                else:
                    self.ctl.assign_external(
                        Function("gcol",
                                [Number(c), Number(int(relative[0])), Number(0)]),
                        True)
                    #log.info(f"gcol({c},{int(relative[0])},0).")
                    self.ctl.assign_external(
                        Function("grow",
                                [Number(c), Number(int(relative[1])), Number(0)]),
                        True)
                    #log.info(f"grow({c},{int(relative[1])},0).")
                    
            # number of violations
            self.ctl.assign_external(Function("num_violations", [Number(num_violations)]), True)
            #log.info(f"num_violations({num_violations}).")
            # dynamic horizon
            #log.info(f"setting dynamic horizon to {dynamic_horizon}")

            self.ctl.assign_external(Function("dynamic_horizon", [Number(dynamic_horizon)]), True)
            # has already eaten a ghost before
            if self.permissive:
                #log.info(f"setting eaten_ghost to {eaten_ghost_before}")
                self.ctl.assign_external(Function("eaten_ghost", [Number(0)]), eaten_ghost_before)

            else: 
                self.ctl.assign_external(Function("eaten_ghost", [Number(0)]), False)



            

    def less_violations_possible(self, state, num_violations, dynamic_horizon,eaten_ghost_before):

        # Reset the clingo window
        self.reset_clingo_externals(state)

        # Set all externals of the currently considered window (see section
        # "Optimization-> Windowing" in the main document for more information)
        # Externals include cell information (walls, ghosts) as well as
        # information about policy preferences
        self.set_clingo_externals(state, num_violations, dynamic_horizon,eaten_ghost_before)

        # solve the LP
        result = self.ctl.solve(on_model=self.on_model)
    
        return result.satisfiable