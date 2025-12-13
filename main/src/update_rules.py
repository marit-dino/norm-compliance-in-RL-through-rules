import hydra
from omegaconf import DictConfig
from legible.env_util import create_environment_and_modelname_for_oftendeeprl
from legible.evaluate_policy import setup_shield, change_action
from legible.shield.shields import RuleChooser 
from legible.create_rules_pacman import string_to_rule
from legible.shield.create_rules_common import turn_rules_to_str
from legible.rule_learning.util import load_model
from legible.feature_and_rule_learn import get_features_and_failure_indication
from gym_pacman_rules.envs.featureExtractors import ExtendedExtractor8, ExtendedExtractor6, ExtendedExtractor7, ExtendedExtractor9, DeepRLCompleteExtractor
from oftendeeprl.env_util import PacmanClingoHelper, get_gym_to_action_names
from pacman_helper_asp import PacmanViolationClingoHelper
import sys, logging, torch
import check_norms
from os import listdir
from os.path import isfile, join
import numpy as np
from collections import deque 

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def setup_update(cfg : DictConfig) -> None:
    norm_descriptor = f"{'_'.join(cfg.norms)}"
    config_str = f"{norm_descriptor}__{str(cfg.asp.horizon)}_{str(cfg.asp.radius)}"

    env, model_name, model_path = create_environment_and_modelname_for_oftendeeprl("norm_guided_dqn", cfg.env.name,
                                                                                    cfg.env.level, config_str, 
                                                                                    cfg.training.steps_initial, cfg.training.steps_norm,
                                                                                    cfg.training.feature_extractor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = env.action_space.n
    action_tensor = torch.tensor(range(num_actions), device=device)

    model_number = get_model_number(cfg, config_str)
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

    update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg)
    update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg)







# adapted from legible
def update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg):
    total_violations = 0
    last_n_violations = deque(maxlen=cfg.asp.horizon+1)
    last_n_states = deque(maxlen=cfg.asp.horizon+1)
    last_n_actions = deque(maxlen=cfg.asp.horizon+1)
    last_n_triggered_rules = deque(maxlen=cfg.asp.horizon+1)
    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    last_n_actions.append(None)
    last_n_violations.append(0)
    feature_extractor = get_feature_extractor(cfg.rules.feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)
    # TODO determine vegetarian/norms
    last_n_states.append(env.unwrapped.game.state)
    asp_helper = PacmanViolationClingoHelper(cfg.asp.horizon, cfg.asp.radius, number_of_ghosts(cfg.env.level), vegetarian=True, num_norms=len(cfg.norms))


    # TODO until convergence etc.
    for i in range(50000):
        total_violations += last_n_violations[-1]
        action, _states = model.predict(obs)
        q_values = policy.q_net(obs_t).squeeze()
        act_logits = q_values        
        raw_state = env.unwrapped.game.state
        obs_features = feature_extractor.getFeatures(raw_state,action)
        obs_flat = np.array([obs_features[j] for j in obs_features.keys()])
        
        if len(shield.get_blocked_actions(obs_flat)) == 4:
            update(last_n_states[-2], rule_chooser, shield, last_n_actions[-2], last_n_triggered_rules[-2], cfg.env, cfg.rules.feature_extractor, feature_extractor, cfg.rules.exclude_features_in_neg_rules)
            i = i-1
            last_n_actions.pop
            last_n_states.pop
            last_n_triggered_rules.pop
            continue

        action, triggered_rules = get_action(model, obs, shield, obs_flat, rule_chooser, cfg.training.algorithm, act_logits, action_tensor, last_n_actions[-1])
        last_n_triggered_rules.append(triggered_rules)

        obs, reward, term, trunc, info = env.step(action)
        last_n_states.append(env.unwrapped.game.state)

        obs_t, vectorized_env = policy.obs_to_tensor(obs)
        obs_t = obs_t.to(action_tensor.device)
        last_n_actions.append(action)
        last_n_violations.append(check_norms.num_violations_detected(cfg.norms, env.unwrapped.game.state))

        if last_n_violations[-1] > 0:
            less_violations_possible = []
            last_n_states_copy = last_n_states.copy()
            for j, state in enumerate(last_n_states):
                log.info(f"step {i+j-cfg.asp.horizon}: \n{state}")
                if j < cfg.asp.horizon:
                    less_violations_possible.append(asp_helper.less_violations_possible(last_n_states_copy.popleft(), last_n_violations[-1], cfg.asp.horizon-j+1))
            print(less_violations_possible)
            # try:
            change_index = len(less_violations_possible) - 1 - less_violations_possible[::-1].index(True)
            update(last_n_states[change_index], rule_chooser, shield, last_n_actions[-1], triggered_rules, cfg.env, cfg.rules.feature_extractor, feature_extractor, cfg.rules.exclude_features_in_neg_rules)
            # except:
            #     log.info("Nothing to update, the number of violations cannot be decreased reliably.")

        if term and reward > 0:
            win = True 
        if term or trunc:
            obs, info = env.reset()
            obs_t, vectorized_env = policy.obs_to_tensor(obs)
            obs_t = obs_t.to(action_tensor.device)
            last_n_states = deque(maxlen=cfg.asp.horizon+1)
            last_n_actions = deque(maxlen=cfg.asp.horizon+1)
            last_n_actions.append(None)

    print(f"TOTAL VIOLATIONS: {total_violations}")


    


def update(state, rule_chooser, shield, action, triggered_rules, env_info, feature_extractor_name, feature_extractor, exclude_features):
    state_features = feature_extractor.getFeatures(state,action)
    if triggered_rules != None:
        categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(env_info.name, env_info.level, feature_extractor_name)
        (pos_rules_triggered, neg_rules_triggered) = triggered_rules
        for rule in neg_rules_triggered + pos_rules_triggered:
            if has_rule_only_categorical_features(rule, categorical_features):
                rule = update_categorical(rule, state_features)
    
    add_neg_rule(state_features, action, shield, rule_chooser, exclude_features)


def add_neg_rule(state_features, action, shield, rule_chooser, exclude_features):
    state_features = [f"f{i}({int(value)})" for i, (key, value) in enumerate(state_features.items()) if i not in exclude_features]
    neg_rule = f"-action({action}) :- "
    for f in state_features:
        neg_rule += f"{f}, "
    neg_rule = neg_rule[:-2] + "."
    cancelable_rule = dict()
    cancelable_rule[neg_rule] = [string_to_rule(neg_rule, False)]
    rules_str = turn_rules_to_str(cancelable_rule)
    shield.cancelable_rules.update(rules_str)
    rule_chooser.rules_list.append(len(rule_chooser.rules_list))
    new_sorted = sorted(shield.cancelable_rules.keys())
    rule_chooser.sorted_cancel_rules = new_sorted
    shield.add_neg_rule(neg_rule)

# def adapt_pos_rule(state_features, rule, shield, rule_chooser, exclude_features):
#     state_features = [f"f{i}({int(value)})" for i, (key, value) in enumerate(state_features.items()) if i not in exclude_features]
#     features_values_rule = rule.strip(".-").split("action(")[1][6:].split(",")
    
#     cancelable_rule = dict()
#     cancelable_rule[neg_rule] = [string_to_rule(neg_rule, False)]
#     shield.cancelable_rules.update(turn_rules_to_str(cancelable_rule))
#     rule_chooser.rules_list.append(len(rule_chooser.rules_list))
#     rule_chooser.sorted_cancel_rules = sorted(shield.cancelable_rules.keys())

def update_categorical(rule, state_features):
    features_values_rule = rule.strip(".-").split("action(")[1][6:].split(",")
    state_features = [f"f{i}({int(value) if isinstance(value, bool) else value})" for i, (key, value) in enumerate(state_features.items())]
    # TODO


# TODO check state instead?
def has_rule_only_categorical_features(rule, categorical_features):
    rule_features = rule.strip(".-").split("action(")[1][6:].split(",")
    for f in rule_features:
        f_num = int(f.strip("f")[:-3])
        if not f_num in categorical_features:
            return False
    return True



def get_action(model, obs, shield, obs_flat, rule_chooser, algo_name, act_logits, action_tensor, last_action):
    action, _states = model.predict(obs)

    triggers,triggered_actions, triggered_rules= shield.does_rule_trigger(obs_flat,rule_chooser)

    if triggers:
        (pos_actions_triggered, neg_actions_triggered) = triggered_actions
        changed_action = change_action(action,pos_actions_triggered,neg_actions_triggered,algo_name,act_logits, action_tensor,
                                        last_action,change_type="favor_cancel")
        if changed_action is not None:
            action = changed_action
    return action, triggered_rules


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
    rule_chooser.set_rules_list(list(range(0,len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))))
    return shield, rule_chooser
    

def number_of_ghosts(level):
    if level.startswith("small"):
        return 2
    else:
        return 4

def get_model_number(cfg, norm_descriptor):
    if cfg.rules.model_number is None:
        model_name = f"norm_guided_dqn__{norm_descriptor}__{cfg.env.name.replace('/', '_')}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_level_{cfg.env.level}_{cfg.training.feature_extractor}"
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
        shields = [f for f in listdir('pickles/shields/uncorr') if f.startswith(shield_name)]
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
    

if __name__ == "__main__":
    setup_update()