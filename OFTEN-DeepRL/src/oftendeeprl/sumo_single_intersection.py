import argparse
import os
import sys
from datetime import datetime

import torch
from stable_baselines3 import DQN

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment

from util import save_model, save_pickle
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from gymnasium import spaces

class IncludeAmbulancesObservationFunction(ObservationFunction):
    """Default observation function in which information about waiting ambulances is included."""

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        ambulances = self.waiting_time_ambulances()

        observation = np.array(phase_id + min_green + density + queue + ambulances, dtype=np.float32)
        # print(observation)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + 2, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + 2, dtype=np.float32),
        )
    
    def waiting_time_ambulances(self):
        env = self.ts.env
        # Ambulances waiting at each edge (exclude internal edges :t_)
        edges = [e for e in env.sumo.edge.getIDList() if not e.startswith(':t_')]
        ambulances_wait_time = []
        # n_t: north -> intersection, t_e: intersection -> east, 
        # t_s: intersection -> south, w_t: from west -> intersection
        for edge in edges: 
            if edge == "t_e" or edge == "t_s":
                continue
            # num_stopped = env.sumo.edge.getLastStepHaltingNumber(
            #     edge)
            # Get all vehicle IDs on this edge
            vehicle_ids = env.sumo.edge.getLastStepVehicleIDs(edge)
            # Filter ambulances
            ambulances = [v for v in vehicle_ids if
                            env.sumo.vehicle.getTypeID(
                                v) == "ambulance"]
            max_wait = -1 # if no ambulance, -1. If there is an ambulance but 
                        # it is not waiting, 0. Otherwise, the number of seconds waiting
            if ambulances:
                # Get waiting times for each ambulance
                wait_times = [env.sumo.vehicle.getWaitingTime(v)
                                for v in ambulances]
                max_wait = int(max(wait_times))
            ambulances_wait_time.append(self.normalize_ambulance_wait_time(max_wait))
        return ambulances_wait_time
    
    @staticmethod
    def normalize_ambulance_wait_time(x):
        # -1 maps to 0
        # inf maps to 1
        if x < 0: # to avoid values close to 0 but not 0
            return 0.0
        else:
            return ((1+np.e)/np.e) * (1/(1+np.exp(-x))) - 1/np.e
    
    @staticmethod
    def denormalize_ambulance_wait_time(x):
        if x == 0:
            return -1
        else:
            return -np.log( ((1+np.e)/np.e) * (1/(x+1/np.e)) -1 )

def create_sumo_env(net_file, route_file, out_csv_name, use_gui, num_seconds, min_green, max_green, delta_time):
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=out_csv_name,
            use_gui=use_gui,
            num_seconds=num_seconds,
            min_green=min_green,
            max_green=max_green,
            delta_time=delta_time,
            observation_class=IncludeAmbulancesObservationFunction,
            single_agent=True,
            reward_fn="queue",
            yellow_time=1,
        )
        
        return env

if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Q-Learning Single-Intersection"""
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.1,
                     required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99,
                     required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05,
                     required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005,
                     required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0,
                     required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10,
                     required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50,
                     required=False, help="Maximum green time.\n")
    prs.add_argument("-deltatime", dest="delta_time", type=int, default=2,
                     required=False, help="Time between simulation steps.\n")
    prs.add_argument("-gui", action="store_true", default=False,
                     help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False,
                     help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False,
                     help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False,
                     help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=1000,
                     required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False,
                     help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1,
                     help="Number of runs.\n")
    prs.add_argument("-t", dest="steps", type=int, default=100_000,
                      help='number of learning steps')
    prs.add_argument("-intersection_type", dest="intersection_type", type=str, default="singleIntersection",
                      help='type of intersection')
    prs.add_argument("-ambulance_prob", dest="ambulance_prob", type=float, default=0.01,
                      help='probability of ambulances')
    prs.add_argument("-flow_north_south_prob", dest="flow_north_south_prob", type=float, default=0.2,
                      help='probability of flow of vehicles from north to south')
    prs.add_argument("-flow_west_east_prob", dest="flow_west_east_prob", type=float, default=0.5,
                      help='probability of flow of vehicles from west to east')
    prs.add_argument("-seed", dest="seed", type=int, default=1337, help='Seed for libusmo.')

    args = prs.parse_args()

    # the number of steps per simulation will be num_seconds/deltatime. By default,
    # 1000/5 = 200
    
    if ("LIBSUMO_AS_TRACI" in os.environ) and args.gui:
        raise Exception('GUI is selected but LIBSUMO_AS_TRACI is set. Please ' \
                'disable GUI or LIBSUMO_AS_TRACI (remove from .bashrc file).')
    
    experiment_time = str(datetime.now()).split(".")[0]
    # out_csv = (f"./gym_sumo/experiments/singleIntersection/{experiment_time}_alpha"
    #            f"{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{
    #            args.decay}")
    out_csv = None

    tb_name = "dqn"
    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None

    algo_name = "dqn"
    env_name = "sumo"

    intersection_type = args.intersection_type
    ambulance_prob = args.ambulance_prob
    flow_north_south_prob = args.flow_north_south_prob
    flow_west_east_prob = args.flow_west_east_prob
    seed = int(args.seed)

    route = f"./gym_sumo/experiments/singleIntersection/{intersection_type}-ambulance{ambulance_prob}-flowns{flow_north_south_prob}-flowwe{flow_west_east_prob}.rou.xml"
    
    level = f"{intersection_type}_{ambulance_prob}_{flow_north_south_prob}_{flow_west_east_prob}_{seed}".replace(".","-")
    
    policy_kwargs_dqn = dict(
        net_arch=[256]*4,
        activation_fn=torch.nn.ReLU,
    )
    policy = "MlpPolicy"
    
    env = make_vec_env( 
            lambda: create_sumo_env(
                net_file=
                "./gym_sumo/experiments/singleIntersection/" \
                    "singleIntersection.net.xml",
                route_file=route,
                out_csv_name=out_csv,
                use_gui=args.gui,
                num_seconds=args.seconds,
                min_green=args.min_green,
                max_green=args.max_green,
                delta_time=args.delta_time,
            ), n_envs = 1, seed=seed)
    
    model = DQN(policy, env, device="cuda", verbose=1, tensorboard_log=tb_path, 
                policy_kwargs=policy_kwargs_dqn, batch_size=256, buffer_size=50_000, 
                exploration_fraction=0.5, gamma = 0.95,gradient_steps=-1
                )

    # Stop training if there is no improvement after more than 20 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=20, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    # The training will stop as soon as the the number of consecutive evaluations without model
    # improvement is greater than 20
    model.learn(args.steps)#, callback=eval_callback)

    model_name = f"pickles/models/{algo_name}_{env_name}_{args.steps}_level_{level}"

    save_model(model_name, model)
