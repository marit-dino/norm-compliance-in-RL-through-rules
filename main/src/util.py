import sys
from os import listdir
from os.path import isfile, join
from gym_pacman_rules.envs.featureExtractors import ExtendedExtractor8, ExtendedExtractor6, ExtendedExtractor7, ExtendedExtractor9, DeepRLCompleteExtractor
from legible.env_util import create_environment_and_modelname_for_oftendeeprl
from legible.rule_learning.util import load_model
import torch


def get_model_number(cfg):
    config_str = f"{cfg.norm.id}__{cfg.asp.horizon}_{cfg.asp.radius}"
    if cfg.rules.model_number is None:
        model_name = f"norm_guided_dqn__{config_str}__{cfg.env.name.replace('/', '_')}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_level_{cfg.env.level}_{cfg.training.feature_extractor}"
        norm_guided_models = [f for f in listdir('./pickles/models') if isfile(join('./pickles/models', f)) 
                            and f.startswith(model_name)]
        if len(norm_guided_models) == 0:
            sys.exit(f"No policy found that matches the provided parameters: {model_name}")
        return sorted(norm_guided_models)[-1].removesuffix(".zip").rsplit("_", 1)[-1]
    else: 
       return cfg.rules.model_number
    
def get_shield_number(cfg):
    if cfg.rules.shield_number is None:
        config_str = f"{cfg.norm.id}__{cfg.asp.horizon}_{cfg.asp.radius}"
        shield_name = f"norm_guided_dqn__{config_str}__{cfg.env.name.replace('/', '_')}_{cfg.env.level}_feat_{cfg.rules.nr_features}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_shield"
        shields = [f for f in listdir('pickles/shields/uncorr') if f.startswith(shield_name) and not "updated" in f]
        if len(shields) == 0:
            sys.exit(f"No shield found that matches the provided parameters: {shield_name}")
        return sorted(shields)[-1].removesuffix(".pkl").rsplit("_", 1)[-1]
    else: 
       return cfg.rules.shield_number
    

def setup_model(cfg):

    config_str = f"{cfg.norm.id}__{cfg.asp.horizon}_{cfg.asp.radius}"
    env, model_name, model_path = create_environment_and_modelname_for_oftendeeprl("norm_guided_dqn", cfg.env.name,
                                                                                    cfg.env.level, config_str, 
                                                                                    cfg.training.steps_initial, cfg.training.steps_norm,
                                                                                    cfg.training.feature_extractor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = env.action_space.n
    action_tensor = torch.tensor(range(num_actions), device=device)

    model_number = get_model_number(cfg)
    model = load_model(model_path + f"_{model_number}","norm_guided_dqn",env=env,exact_match=True)
    return env, model, model_name, action_tensor

def order_feature_indices(shield, env, feature_extractor):
    obs, info = env.reset()
    feature_extractor = get_feature_extractor(feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)
    features = feature_extractor.getFeatures(env.unwrapped.game.state,None,False)
    ordered_features = sorted(list(features.items()),key=lambda x: x[0])

    feature_to_index = {
        name: idx for idx, name in enumerate(features)
    }
    ordered_feature_indices = [feature_to_index[fv[0]] for fv in ordered_features]
    shield.feature_indices = ordered_feature_indices


def get_feature_extractor(feature_extractor, height, width):
    if feature_extractor == "extended-6":
        return ExtendedExtractor6(height=height,width=width)
    elif feature_extractor == "extended-7":
        return ExtendedExtractor7(height=height,width=width)
    elif feature_extractor == "extended-8":
        return ExtendedExtractor8(height=height,width=width)
    elif feature_extractor == "extended-9":
        return ExtendedExtractor9(height=height,width=width)
    elif feature_extractor == "complete" :
        return DeepRLCompleteExtractor(height=height, width=width)
    

