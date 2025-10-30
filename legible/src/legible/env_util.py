import copy

from legible.train_pacman import create_pacman_env


def create_environment_and_modelname(algo_name, env_name, mode, steps, feature_extractor=None):
    if "Pacman" in env_name:
        model_name = f"{algo_name}_{env_name.replace('/', '_')}_{steps}_level_{mode}_{feature_extractor}"
    else:
        model_name = f"{algo_name}_{env_name.replace('/', '_')}_{steps}"
    model_path = f"pickles/models/{model_name}"
    if "highway" in env_name or "intersection" in env_name or "roundabout" in env_name or "merge" in env_name:
        loc_env_kwargs = copy.deepcopy(env_kwargs)
        loc_env_kwargs["id"] = env_name

        env = create_hw_env(**loc_env_kwargs)
    elif "MiniGrid" in env_name:
        env = create_mg_env(env_name)
    else:
        env = create_pacman_env(env_name, feature_extractor,mode,scale=algo_name=="ppo", render_mode="none")
    return env, model_name, model_path
