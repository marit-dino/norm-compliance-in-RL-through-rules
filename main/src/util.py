import sys
from os import listdir
from os.path import isfile, join
from gym_pacman_rules.envs.featureExtractors import ExtendedExtractor8, ExtendedExtractor6, ExtendedExtractor7, ExtendedExtractor9, DeepRLCompleteExtractor
from legible.rule_learning.util import save_pickle
from legible.shield.shields import RuleChooser 
from legible.create_rules_pacman import string_to_rule




class RuleSnapshot:
    enforceable_rules = dict()
    cancelable_rules = dict()
    rule_indices = []

    def __init__(self, enforceable_rules, cancelable_rules):
        self.enforceable_rules = enforceable_rules 
        self.cancelable_rules = cancelable_rules  
        self.rule_indices = range(0, len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))


def get_model_number(cfg):
    norm_descriptor = f"{'_'.join(cfg.norms)}"
    config_str = f"{norm_descriptor}__{str(cfg.asp.horizon)}_{str(cfg.asp.radius)}"

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
        shield_name = f"norm_guided_dqn_{cfg.env.name.replace('/', '_')}_{cfg.env.level}_feat_{cfg.rules.nr_features}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_shield"
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
    

# adapted from legible
def set_rules(shield):
    enforceable_rules = dict()
    cancelable_rules = dict()
    for pos_rule in shield.pos_rules_list:
        if len(pos_rule.strip()) == 0:
            continue
        enforceable_rules[pos_rule] = [string_to_rule(pos_rule, True)]
    for neg_rule in shield.neg_rules_list:
        if len(neg_rule.strip()) == 0:
            continue
        cancelable_rules[neg_rule] = [string_to_rule(neg_rule, True)]
    shield.enforceable_rules = enforceable_rules
    shield.cancelable_rules = cancelable_rules
    rule_chooser = RuleChooser(shield)
    rule_chooser.set_rules_list(list(range(0,len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))))
    return shield, rule_chooser
    
def save_rule_set(shield, cfg):
    shield_name = f"pickles/shields/uncorr/"\
                  f"norm_guided_dqn_{cfg.env.name.replace('/','_')}_{cfg.env.level}_feat_{cfg.rules.nr_features}_"\
                  f"{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_shield_updated"

    if cfg.rules.shield_number is None:
        save_pickle(shield_name,shield)
    else:
        shield_name = f"{shield_name}_updated_{cfg.rules.shield_number}.pkl"
        save_pickle(shield_name,shield,exact_match=True)
