import clingo
from clingo import Number, Function

from gym_gardener.utils import get_current_time_ms
from sb3_ext.clingoHelper import ClingoHelper

class SumoHelper(ClingoHelper):

    def __init__(self, horizon):
        self.horizon = horizon
        self.next = None
        self.penalty = 0
    
    def setup(self, clingo_static, excluded):
        tl = len(clingo_static['tls'])
        edges = len(clingo_static['edges'])

        self.ctl = clingo.Control(
            ["-c", f"horizon={self.horizon}",
             "-c", f"tl={tl}",
             "-c", f"edges={edges}",
             ])
        for tl in clingo_static['tls']:
            for edge_phase in clingo_static['tls'][tl]['edge_phases']:
                edge_id = clingo_static['edges'][edge_phase]
                tl_id = clingo_static['tls'][tl]['id']
                phase_id = clingo_static['tls'][tl]['edge_phases'][edge_phase]
                self.ctl.add("base", [], f"""
                        green_edge({edge_id}, T) :- tl_phase({tl_id}, {phase_id}, T).
                    """)

        self.ctl.load('./gym_sumo/program.lp')
        self.ctl.ground([("base", [])], context=self)

        self.clingo_static = clingo_static
        self.excluded = excluded

    def get_relevant_states(self, state, obs):
        return [[obs,"curr"]]

    def on_model(self, m):
        self.save_solution(m)
        self.show = " ".join([str(i) for i in m.symbols(shown=True)])
        # print("Answer:\n{}".format(self.show))

    def save_solution(self, m):
        self.penalty = 0
        self.next = {}
        for sym in m.symbols(shown=True):
            if sym.name == "tl_act":
                tl_id = sym.arguments[0].number
                if sym.arguments[2].number == 0:
                    self.next[tl_id] = sym.arguments[1].number
                continue

    def reset_clingo_externals(self):
        for tl in self.clingo_static['tls']:
            tl_id = self.clingo_static['tls'][tl]['id']
            for phase in self.clingo_static['tls'][tl]['phases']:
                self.ctl.assign_external(Function("tl_phase", [
                    Number(tl_id), Number(self.clingo_static['tls'][tl]['phases'][phase]), Number(0)
                ]), False)
                # self.ctl.assign_external(Function("tl_pref", [
                #     Number(tl_id), Number(self.clingo_static['tls'][tl]['phases'][phase]),Number(0)
                # ]), False)
        for edge in self.clingo_static['edges']:
            for i in range(self.horizon):
                self.ctl.assign_external(Function("amb_edge", [
                    Number(self.clingo_static['edges'][edge]), Number(i),
                    Number(0)
                ]), False)
        
        # reset actions
        # TODO: add tls
        for i in range(2):
            self.ctl.assign_external(Function("tl_pref", [
                    Number(0),Number(i),Number(0)
                ]), False)


    def set_clingo_externals(self, clingo_dynamic, actionValuePairs):
        for tl in clingo_dynamic['tls']:
            tl_id = self.clingo_static['tls'][tl]['id']
            self.ctl.assign_external(Function("tl_phase", [
                Number(tl_id),Number(clingo_dynamic['tls'][tl]['phase']),Number(0)
            ]), True)
            # self.ctl.assign_external(Function("tl_pref", [
            #     Number(tl_id),Number(clingo_dynamic['actions'][tl]),Number(0)
            # ]), True)

        for edge in clingo_dynamic['edges']:
            wait_time = clingo_dynamic['edges'][edge]
            if wait_time > self.horizon:
                wait_time = self.horizon
            self.ctl.assign_external(Function("amb_edge", [
                Number(self.clingo_static['edges'][edge]), Number(wait_time),
                Number(0)
            ]), True)

        # encode the policy preferences
        sortedActionValuePairs = sorted(actionValuePairs, key=lambda x: x[1],
                                        reverse=True)

        # TODO: add tls
        self.ctl.assign_external(Function("tl_pref", [
                Number(0),Number(sortedActionValuePairs[0][0]),Number(0)
            ]), True)

    def get_action(self, state, actionValuePairs):
        self.next = None
        actionValuePairs = actionValuePairs["curr"]

        clingo_dynamic = self.get_clingo_dynamic(state['env'])

        # Reset the clingo window
        self.reset_clingo_externals()

        # Set all externals of the currently considered window (see section
        # "Optimization-> Windowing" in the main document for more information)
        # Externals include cell information (walls, ghosts) as well as
        # information about policy preferences
        self.set_clingo_externals(clingo_dynamic, actionValuePairs)

        # solve the LP
        self.ctl.solve(on_model=self.on_model)

        assert len(self.next) == 1, "length of the next action is different than 1. Not yet implemented"

        return [self.next[x] for x in self.next][0]


    def get_clingo_dynamic(self, env):
        # Traffic light status
        clingo_dynamic = {'tls': {}}
        edges_wait = {}
        for tls_id in env.ts_ids:
            if tls_id in self.excluded:
                continue
            clingo_dynamic['tls'][tls_id] = {}
            state = env.sumo.trafficlight.getRedYellowGreenState(
                tls_id)
            clingo_dynamic['tls'][tls_id]['phase'] = \
            self.clingo_static['tls'][tls_id]['phases'][state]

        # Ambulances waiting at each edge (exclude internal edges :t_)
        for edge in self.clingo_static['edges']:
            # Get all vehicle IDs on this edge
            vehicle_ids = env.sumo.edge.getLastStepVehicleIDs(edge)
            # Filter ambulances
            ambulances = [v for v in vehicle_ids if
                          env.sumo.vehicle.getTypeID(
                              v) == "ambulance"]
            max_wait = -1
            if ambulances:
                # Get waiting times for each ambulance
                wait_times = [env.sumo.vehicle.getWaitingTime(v)
                              for v in ambulances]
                max_wait = int(max(wait_times))
            edges_wait[edge] = max_wait
        clingo_dynamic['edges'] = edges_wait
        return clingo_dynamic

    @staticmethod
    def get_clingo_static(env, excluded):
        clingo_static = {'tls': {}}
        edge_ids = {}
        tls_num = 0
        edge_id = 0
        for tls_id in env.ts_ids:
            controlled_lanes = env.sumo.trafficlight.getControlledLanes(
                tls_id)
            for lane in controlled_lanes:
                edge = lane.rsplit('_', 1)[0]
                if edge not in edge_ids:
                    edge_ids[edge] = edge_id
                    edge_id += 1

            if tls_id in excluded:
                continue
            clingo_static['tls'][tls_id] = {}
            clingo_static['tls'][tls_id]['id'] = tls_num
            edge_phases = {}
            tls_num += 1
            phase_ids = {}
            phase_id = 0
            static_phases = env.sumo.trafficlight.getAllProgramLogics(tls_id)
            for program in static_phases:
                if program.programID == '0':
                    print(f"Program ID: {program.programID}")
                    for phase in program.phases:
                        for idx, lane in enumerate(controlled_lanes):
                            if phase.state[idx] == 'G':
                                edge = lane.rsplit('_', 1)[0]
                                if edge not in edge_phases:
                                    edge_phases[edge] = phase_id
                        if 'y' not in phase.state:
                            print(
                                f"Duration: {phase.duration}, State: {phase.state}, ID: {phase_id}")
                            phase_ids[phase.state] = phase_id
                            phase_id += 1
            clingo_static['tls'][tls_id]['phases'] = phase_ids
            clingo_static['tls'][tls_id]['edge_phases'] = edge_phases
        clingo_static['edges'] = edge_ids

        return clingo_static