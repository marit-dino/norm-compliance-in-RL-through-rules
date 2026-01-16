import sys
from os import listdir
from os.path import isfile, join
from gym_pacman_rules.envs.featureExtractors import ExtendedExtractor8, ExtendedExtractor6, ExtendedExtractor7, ExtendedExtractor9, DeepRLCompleteExtractor



def get_model_number(cfg):
    norm_descriptor = f"{'_'.join(cfg.norms)}"
    config_str = f"{norm_descriptor}__{cfg.asp.horizon}_{cfg.asp.radius}"

    if cfg.rules.model_number is None:
        model_name = f"norm_guided_dqn__{config_str}__{cfg.env.name.replace('/', '_')}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_level_{cfg.env.level}_{cfg.training.feature_extractor}"
        norm_guided_models = [f for f in listdir('./pickles/models') if isfile(join('./pickles/models', f)) 
                            and f.startswith(model_name)]
        if len(norm_guided_models) == 0:
            sys.exit("No policy found that matches the provided parameters.")
        return sorted(norm_guided_models)[-1].removesuffix(".zip").rsplit("_", 1)[-1]
    else: 
       return cfg.rules.model_number
    
def get_shield_number(cfg):
    if cfg.rules.shield_number is None:
        shield_name = f"norm_guided_dqn_{cfg.env.name.replace('/', '_')}_{cfg.env.level}_{cfg.asp.horizon}_{cfg.asp.radius}_feat_{cfg.rules.nr_features}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_shield"
        shields = [f for f in listdir('pickles/shields/uncorr') if f.startswith(shield_name) and not "updated" in f]
        if len(shields) == 0:
            sys.exit("No shield found that matches the provided parameters.")
        return sorted(shields)[-1].removesuffix(".pkl").rsplit("_", 1)[-1]
    else: 
       return cfg.rules.shield_number
    

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
    

