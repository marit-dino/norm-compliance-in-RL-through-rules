import torch

from .sb3_ext.pacman_helper import PacmanClingoHelper
from .train_pacman import create_pacman_env
from stable_baselines3.common.env_util import make_vec_env

def get_gym_to_action_names(env_name) :
    if "Pacman" in env_name:
        return {
            0: 'North',
            1: 'South',
            2: 'East',
            3: 'West',
        }
    elif "garden" in env_name:
        return {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }
    elif "sumo" in env_name:
        return {
            0: 0,
            1: 1,
        }
    else:
        raise Exception("Unsupported")

def get_policy_kwargs_dqn(env_name):
    if "Pacman" in env_name:
        return dict(
            net_arch=[512,256,128,64],
            activation_fn= torch.nn.ReLU )
    elif "garden" in env_name:
        from gym_gardener.feature_extractors import MinigridFeaturesExtractor
        return dict(
            features_extractor_class=MinigridFeaturesExtractorNonDict,
            features_extractor_kwargs=dict(features_dim=512,view_size=9),
        )
    elif "sumo" in env_name:
        return dict(
            net_arch=[256]*4,
            activation_fn=torch.nn.ReLU)
    else:
        raise Exception("Unsupported")

def get_garden_level_name(level):
    return f"{level[0]}_{level[1]}_{level[2]}_{level[3]}".replace(".","-")

def get_sumo_level_name(level):
    return f"{level[0]}_{level[1]}_{level[2]}_{level[3]}_{level[4]}".replace(".","-")

def create_environment_and_modelname(algo_name, env_name, mode, steps, feature_extractor=None):
    if "Pacman" in env_name:
        model_name = f"{algo_name}_{env_name.replace('/', '_')}_{steps}_level_{mode}_{feature_extractor}"
        env = create_pacman_env(env_name, feature_extractor, mode, scale="dqn-n" in algo_name, render_mode="none")
    elif "garden" in env_name:
        (instSize,pct_walls,pct_plants,pct_frogs,seed) = mode
        level_name = get_garden_level_name(mode)
        model_name = f"{algo_name}_{env_name.replace('/', '_')}_{steps}_level_{level_name}"
        env = create_gardener_env(instSize,pctg_walls=pct_walls, pctg_phenomena=pct_plants, pctg_frogs=pct_frogs,
                                  view_size=9,instance_seed=seed)
    elif "sumo" in env_name:
        model_name = f"{algo_name}_{env_name}_{steps}_level_{mode}"
        intersection_type, ambulance_prob, flow_north_south_prob, flow_west_east_prob, seed = mode.replace("-",".").split("_")
        mode.split("_")[-1]
        # to avoid the problem with the seed in libsumo, we fix the
        # seed using a 1-dimension vectorized environment and then
        # select the sumo environment that was created
        env = make_vec_env( 
            lambda: create_sumo_env(
            net_file="./gym_sumo/experiments/singleIntersection/singleIntersection.net.xml",
            route_file=f"./gym_sumo/experiments/singleIntersection/{intersection_type}-ambulance{ambulance_prob}-flowns{flow_north_south_prob}-flowwe{flow_west_east_prob}.rou.xml",
            out_csv_name=None,
            use_gui=False,
            num_seconds=1000,
            min_green=10,
            max_green=50,
            delta_time=2,
        ), n_envs=1, seed = int(seed))
        env = env.envs[0].env
    else:
        raise Exception("Not supported")
    model_path = f"pickles/models/{model_name}"

    return env, model_name, model_path

def get_norm_helper(env_name,arg, level):
    if "Pacman" in env_name:
        norm_tuple_str = arg.replace("--norm","")
        split_str = norm_tuple_str.split("-")
        horizon = int(split_str[0])
        radius = int(split_str[1])
        vegetarian = split_str[2] == "True"
        ghosts = 4 if "original" in level else 2
        norm_descriptor = "vegetarian" if vegetarian else "vegan"
        norm_descriptor += f"_{horizon}_{radius}"
        norm_helper = PacmanClingoHelper(horizon, radius, ghosts, vegetarian=vegetarian)
    elif "garden" in env_name:
        norm_tuple_str = arg.replace("--norm","")
        split_str = norm_tuple_str.split("-")
        horizon = int(split_str[0])
        radius = int(split_str[1])
        norm_descriptor = f"_{horizon}_{radius}"
        norm_helper = GardenerHelper(horizon,radius)
    elif "sumo" in env_name:
        norm_tuple_str = arg.replace("--norm","")
        split_str = norm_tuple_str.split("-")
        horizon = int(split_str[0])
        norm_descriptor = f"_{horizon}"
        norm_helper = SumoHelper(horizon)
    else:
        raise Exception("Unsupported")

    return norm_descriptor, norm_helper
