import hydra
from omegaconf import DictConfig
from legible.env_util import create_environment_and_modelname_for_oftendeeprl
from legible.evaluate_policy import setup_shield, eval_single_eps, EvalStats, change_action
from legible.shield.shields import RuleChooser 
from legible.create_rules_pacman import string_to_rule
from legible.shield.create_rules_common import turn_rules_to_str
from legible.rule_learning.util import load_model
from gym_pacman_rules.envs.featureExtractors import ExtendedExtractor8, ExtendedExtractor6, ExtendedExtractor7, ExtendedExtractor9, DeepRLCompleteExtractor
import torch
import sys
from os import listdir
from os.path import isfile, join
import numpy as np

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def update_rules(cfg : DictConfig) -> None:
    norm_descriptor = cfg.norm + "_" + str(cfg.asp.horizon) + "_" + str(cfg.asp.radius)

    env, model_name, model_path = create_environment_and_modelname_for_oftendeeprl("norm_guided_dqn", cfg.env.name,
                                                                                    cfg.env.level, norm_descriptor, 
                                                                                    cfg.training.steps_initial, cfg.training.steps_norm,
                                                                                    cfg.training.feature_extractor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = env.action_space.n
    action_tensor = torch.tensor(range(num_actions), device=device)

    model_number = get_model_number(cfg, norm_descriptor)
    model = load_model(model_path + f"_{model_number}","norm_guided_dqn",env=env,exact_match=True)
    shield_number = get_shield_number(cfg)

    #TODO what is difference between uncorr and improved?
    shield = setup_shield(cfg.env.name, cfg.env.level, cfg.training.steps_initial, cfg.rules.nr_features, False, False, exact_model_number=shield_number,
                           steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn")

    if shield is None:
        sys.exit("Could not load shield, check if it exists.")

    shield, rule_chooser = set_rules(shield)
    assert hasattr(shield, 'enforceable_rules')
    assert hasattr(shield, 'cancelable_rules')

    # TODO fix names
    update(env, model, action_tensor, shield, rule_chooser, cfg.training.algorithm, cfg.rules.feature_extractor)




# adapted from legible
def update(env, model, action_tensor, shield, rule_chooser, algo_name, feature_extractor):
    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    last_action = None
    feature_extractor = get_feature_extractor(feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)

    # TODO until convergence etc.
    for i in range(500):
        #print(i)
        action, _states = model.predict(obs)
        q_values = policy.q_net(obs_t).squeeze()
        act_logits = q_values        
        raw_state = env.unwrapped.game.state
        obs_features = feature_extractor.getFeatures(raw_state,action)
        obs_flat = np.array([obs_features[j] for j in obs_features.keys()])
        triggers,triggered= shield.does_rule_trigger(obs_flat,rule_chooser)

        if triggers:
            (pos_triggered, neg_triggered) = triggered
            changed_action = change_action(action,pos_triggered,neg_triggered,algo_name,act_logits, action_tensor,
                                            last_action,change_type="favor_cancel")
            if changed_action is not None:
                action = changed_action

        obs, reward, term, trunc, info = env.step(action)

        obs_t, vectorized_env = policy.obs_to_tensor(obs)
        obs_t = obs_t.to(action_tensor.device)
        last_action = action

        violated_norm(raw_state, env.unwrapped.game.state)

        if term and reward > 0:
            win = True 
        if term or trunc:
            obs, info = env.reset()
            obs_t, vectorized_env = policy.obs_to_tensor(obs)
            obs_t = obs_t.to(action_tensor.device)
            last_action = None


def violated_norm(prev, curr):
    # TODO find where number of times eaten is stored
    return False




# adapted from legible
def set_rules(shield):
    enforceable_rules = dict()
    cancelable_rules = dict()
    for pos_rule in shield.pos_rules_list:
        if len(pos_rule.strip()) == 0:
            continue
        enforceable_rules[pos_rule] = [string_to_rule(pos_rule)]
    for neg_rule in shield.neg_rules_list:
        if len(neg_rule.strip()) == 0:
            continue
        cancelable_rules[neg_rule] = [string_to_rule(neg_rule)]
    shield.enforceable_rules = turn_rules_to_str(enforceable_rules) if len(enforceable_rules) > 0 else dict()
    shield.cancelable_rules = turn_rules_to_str(cancelable_rules) if len(cancelable_rules) > 0 else dict()
    rule_chooser = RuleChooser(shield)
    rule_chooser.set_rules_list(list(range(0, len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))))
    return shield, rule_chooser
    

def get_model_number(cfg, norm_descriptor):
    if cfg.rules.model_number is None:
        model_name = f"norm_guided_dqn__{norm_descriptor}__{cfg.env.name.replace('/', '_')}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_level_{cfg.env.level}_{cfg.training.feature_extractor}"
        norm_guided_models = [f for f in listdir('../pickles/models') if isfile(join('../pickles/models', f)) 
                            and f.startswith(model_name)]
        if len(norm_guided_models) == 0:
            sys.exit("No policy found that matches the provided parameters.")
        return sorted(norm_guided_models)[-1].removesuffix(".zip").rsplit("_", 1)[-1]
    else: 
       return cfg.rules.model_number
    
def get_shield_number(cfg):
    if cfg.rules.shield_number is None:
        shield_name = f"norm_guided_dqn_{cfg.env.name.replace('/', '_')}_{cfg.env.level}_feat_{cfg.rules.nr_features}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_shield"
        shields = [f for f in listdir('pickles/shields/uncorr') if f.startswith(shield_name)]
        if len(shields) == 0:
            print(shield_name)
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

if __name__ == "__main__":
    update_rules()