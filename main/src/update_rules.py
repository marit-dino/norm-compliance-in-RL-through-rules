import hydra
from omegaconf import DictConfig
from util import RuleSnapshot
from legible.env_util import create_environment_and_modelname_for_oftendeeprl
from legible.evaluate_policy import setup_shield, change_action
from legible.shield.shields import RuleChooser 
from legible.create_rules_pacman import string_to_rule
from legible.shield.create_rules_common import turn_rules_to_str
from legible.rule_learning.util import load_model
from legible.feature_and_rule_learn import get_features_and_failure_indication
from gym_pacman_rules.envs.featureExtractors import ExtendedExtractor8, ExtendedExtractor6, ExtendedExtractor7, ExtendedExtractor9, DeepRLCompleteExtractor, features_dict_to_array
from oftendeeprl.env_util import PacmanClingoHelper, get_gym_to_action_names
from pacman_helper_asp import PacmanViolationClingoHelper
import sys, logging, torch
import check_norms
from os import listdir
from os.path import isfile, join
import numpy as np
import copy
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

    obs, info = env.reset()
    feature_extractor = get_feature_extractor(cfg.rules.feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)
    features = feature_extractor.getFeatures(env.unwrapped.game.state,None)
    ordered_features = sorted(list(features.items()),key=lambda x: x[0])
    #features = features_dict_to_array(features)
    feature_to_index = {
        name: idx for idx, name in enumerate(features)
    }
    ordered_feature_indices = [feature_to_index[fv[0]] for fv in ordered_features]

    shield.feature_indices = ordered_feature_indices
    shield, rule_chooser = set_rules(shield)
    assert hasattr(shield, 'enforceable_rules')
    assert hasattr(shield, 'cancelable_rules')

    update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg)
    update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg)





# adapted from legible
def update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg):
    # TODO move a part of this to  a setup method
    total_violations = 0
    last_n_violations = deque(maxlen=cfg.asp.horizon+1)
    last_n_states = deque(maxlen=cfg.asp.horizon+1)
    last_n_actions = deque(maxlen=cfg.asp.horizon+1)
    last_n_triggered_rules = deque(maxlen=cfg.asp.horizon+1)
    prev_env_states = deque(maxlen=cfg.asp.horizon+1)
    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    last_n_actions.append(None)
    last_n_violations.append(0)
    feature_extractor = get_feature_extractor(cfg.rules.feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)
    # TODO determine vegetarian/norms
    last_n_states.append(copy.deepcopy(env.unwrapped.game.state))
    prev_env_states.append(env.unwrapped.save_state())
    asp_helper = PacmanViolationClingoHelper(cfg.asp.horizon, cfg.asp.radius, number_of_ghosts(cfg.env.level), vegetarian=True, num_norms=len(cfg.norms))

    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )

    # TODO until convergence etc. / based on number of episodes
    for i in range(25000):
        if i % 1000 == 0:
            print(i)
        total_violations += last_n_violations[-1]
        action, _states = model.predict(obs)
        q_values = policy.q_net(obs_t).squeeze()
        act_logits = q_values        
        obs_rules = features_dict_to_array(feature_extractor.getFeatures(last_n_states[-1],action))

        
        if len(shield.get_blocked_actions(obs_rules)) == 4:
            all_actions_blocked(obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_triggered_rules, last_n_violations, prev_env_states, env, feature_extractor, cfg)
                    
        action, triggered_rules = get_action(model, obs, obs_rules, shield, last_n_states[-1],last_n_actions[-1], feature_extractor, rule_chooser, rules_snapshot, cfg.training.algorithm, act_logits, action_tensor)
        last_n_triggered_rules.append(triggered_rules)
        
        obs, reward, term, trunc, info = env.step(action)
        last_n_states.append(copy.deepcopy(env.unwrapped.game.state))
        last_n_actions.append(action)
        last_n_violations.append(check_norms.num_violations_detected(cfg.norms, last_n_states[-1]))
        prev_env_states.append(env.unwrapped.save_state())

        obs_t, vectorized_env = policy.obs_to_tensor(obs)
        obs_t = obs_t.to(action_tensor.device)

        if last_n_violations[-1] > 0:
            less_violations_possible = []
            last_n_states_copy = last_n_states.copy()
            for j, state in enumerate(last_n_states):
                log.info(f"step {i+j-cfg.asp.horizon}: \n{state}")
                if j < cfg.asp.horizon:
                    less_violations_possible.append(asp_helper.less_violations_possible(last_n_states_copy.popleft(), sum(last_n_violations), cfg.asp.horizon-j+1))
            print(less_violations_possible)
            if True in less_violations_possible:
                prev_state, prev_action, prev_triggered_rules, prev_obs = backtrack(last_n_actions, last_n_states, last_n_triggered_rules, prev_env_states, env, last_n_violations, feature_extractor, violation=False)
                rules_snapshot = update(last_n_states[-1], prev_obs, rule_chooser, shield, prev_action, prev_triggered_rules, cfg.env, cfg.rules.feature_extractor, feature_extractor, cfg.rules.exclude_features_in_neg_rules)
                continue
            else:
                log.info("Nothing to update, the number of violations cannot be decreased reliably.")

        if term and reward > 0:
            win = True 
        if term or trunc:
            obs, info = env.reset()
            obs_t, vectorized_env = policy.obs_to_tensor(obs)
            obs_t = obs_t.to(action_tensor.device)
            last_n_states = deque(maxlen=cfg.asp.horizon+1)
            last_n_actions = deque(maxlen=cfg.asp.horizon+1)
            last_n_actions.append(None)
            last_n_states.append(env.unwrapped.game.state)


    log.info(f"Total Violations: {total_violations}")


def backtrack(last_n_actions, last_n_states, last_n_triggered_rules, prev_env_states, env, last_n_violations, feature_extractor, violation=True):
    env.unwrapped.load_state(prev_env_states[-2])   
    prev_env_states.pop()
    if violation:
        last_n_violations.pop()
    prev_obs =  features_dict_to_array(feature_extractor.getFeatures(state=last_n_states[-2], action=last_n_actions[-1]))
    return last_n_states.pop(), last_n_actions.pop(), last_n_triggered_rules.pop(), prev_obs



def all_actions_blocked(obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_triggered_rules, last_n_violations, prev_env_states, env, feature_extractor, cfg):
    triggers,triggered_actions, triggered_rules = shield.does_rule_trigger(obs_rules,rule_chooser,rules_snapshot)
    (pos_rules_triggered, neg_rules_triggered) = triggered_rules
    neg_actions = [r[0].rule_head.action for r in neg_rules_triggered]
    print(neg_actions)
    blocked_actions = set(neg_actions)
    actions_blocked_by_created_rules = {a: True for a in blocked_actions}

    for r in neg_rules_triggered:
        if r[0].mined == True:
            actions_blocked_by_created_rules[r[0].rule_head.action] = False
        print(r[0])

    if False not in list(actions_blocked_by_created_rules.values()):
        log.info(f"Backtracking, all actions are blocked by created rules")
        prev_state, prev_action, prev_triggered_rules, prev_obs = backtrack(last_n_actions, last_n_states, last_n_triggered_rules, prev_env_states, env, last_n_violations, feature_extractor, violation=False)
        # TODO what if we backtrack more than the deque is long? (can this even happen?)
        update(last_n_states[-1], prev_obs, rule_chooser, shield, last_n_actions[-1], last_n_triggered_rules[-1], cfg.env, cfg.rules.feature_extractor, feature_extractor, cfg.rules.exclude_features_in_neg_rules)
        print(f"Now in \n {env.unwrapped.game.state}")
    else:
        log.info(f"All actions are blocked, adapting a mined rule")
        mined_neg_rule = list(filter(lambda r: r[0].mined == True, neg_rules_triggered))[0][0]
        print(mined_neg_rule)
        print(feature_extractor.getFeatures(last_n_states[-1],last_n_actions[-1]))
        state_features = feature_extractor.getFeatures(last_n_states[-1],last_n_actions[-1])
        rules_snapshot = adapt_rules_based_on_state(mined_neg_rule,state_features, shield, rule_chooser, cfg)
    sys.exit()


def adapt_rules_based_on_state(mined_rule,state_features, shield, rule_chooser, cfg):
    categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(cfg.env.name, cfg.env.level, cfg.rules.feature_extractor)
    if only_enumerable_features_difference(mined_rule, cfg.rules.features_with_enumerable_values, categorical_features, state_features, shield):
        rules_snapshot = add_differing_enumerable_features(mined_rule, state_features, categorical_features, cfg.rules.features_with_enumerable_values, cfg.env.level, shield, rule_chooser, cfg.rules.exclude_features_in_neg_rules)
        return rules_snapshot
    elif mined_rule.polarity == False:
        rules_snapshot = remove_rule(mined_rule, shield, rule_chooser)
        return rules_snapshot
    elif mined_rule.polarity == True:
        rules_snapshot = add_neg_rule(features_dict_to_array(state_features), mined_rule.rule_head.action, shield, rule_chooser, cfg.rules.exclude_features_in_neg_rules)
        rules_snapshot = remove_rule(mined_rule, shield, rule_chooser)
        return rules_snapshot

    # TODO does this make sense?

def only_enumerable_features_difference(rule, enumerable_features, categorical_features, state_features, shield):
    features_values = shield.discretize(features_dict_to_array(state_features)).astype(int)
    feature_facts = {
        fi: value
        for fi, value in zip(shield.feature_indices, features_values)
    }
    print("conditions: ", rule.rule_body.conditions)
    print("feature facts keys: ", list(feature_facts.keys()))
    occurring_features = [f.feature for f in rule.rule_body.conditions]
    print("occurring features in rule: ", occurring_features)
    differing_features = [f for f in list(feature_facts.keys()) if f not in occurring_features]
    print("feature facts:", feature_facts)
    print("differing features: ")
    for f in differing_features:
        print(f"feature {f.feature} : rule value {f.valuation}")

    for f in differing_features:
        if not f.feature in categorical_features and f.feature not in enumerable_features:
            return False
    return True


def add_differing_enumerable_features(mined_rule,state_features, categorical_features, enumerable_features, level, shield, rule_chooser, exclude_features_in_neg_rules):
    feature_facts = {
        fi: value
        for fi, value in zip(shield.feature_indices, features_dict_to_array(state_features))
    }
    adapted_rule = copy.deepcopy(mined_rule)
    feature_indices_in_rule = [f.feature for f in mined_rule.rule_body.conditions]
    adapted_rules = [adapted_rule]
    for fi in feature_facts.keys():
        if fi in exclude_features_in_neg_rules or fi in feature_indices_in_rule:
            continue
        if fi in categorical_features:
            for i,r in enumerate(adapted_rules):
                adapted_rules[i] = r.add_feature(fi, abs(feature_facts[fi]-1))
        elif fi in enumerable_features:
            for v in range(0, number_of_ghosts(level)+1):
                if v != feature_facts[fi]:
                    for i,r in enumerate(adapted_rules):
                        #TODO fix?
                        adapted_rules.append(r.add_feature(fi, v))
                        adapted_rules.pop(i)


    log.info(f"State features used to adapt the rule:{feature_facts}")
    log.info(f"Original rule: {str(mined_rule)}")
    log.info(f"Adapted rules: {[str(r) for r in adapted_rules]}")
    rules_snapshot = add_rule(adapted_rule, shield, rule_chooser)
    return rules_snapshot
       


def update(state, obs_rules, rule_chooser, shield, action, triggered_rules, env_info, feature_extractor_name, feature_extractor, exclude_features):
    if triggered_rules != None:
        categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(env_info.name, env_info.level, feature_extractor_name)
        (pos_rules_triggered, neg_rules_triggered) = triggered_rules
        # TODO for rule in neg_rules_triggered + pos_rules_triggered:
            # if has_rule_only_categorical_features(rule[0], categorical_features):
            #     state_features = feature_extractor.getFeatures(state,action)
            #     rule[0] = update_categorical(rule[0], state_features)
    return add_neg_rule(obs_rules, action, shield, rule_chooser, exclude_features)


def remove_rule(rule, shield, rule_chooser):
    if rule.polarity == False:
        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.pop()
        cancelable_rules_copy = shield.cancelable_rules.copy()
        cancelable_rules_copy.pop(str(rule))
        new_sorted = sorted(cancelable_rules_copy.keys())

        shield.remove_neg_rule(str(rule))
        shield.cancelable_rules.pop(str(rule))
        rule_chooser.set_rules_list(rule_chooser_rules_copy)
        rule_chooser.sorted_cancel_rules = new_sorted

        log.info(f"Removed rule: {rule}")

        rules_snapshot = RuleSnapshot(
            enforceable_rules = shield.enforceable_rules,
            cancelable_rules = cancelable_rules_copy,
        )
        return rules_snapshot
    
    else:
        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.remove(-1)
        enforcable_rules_copy = shield.enforceable_rules.copy()
        enforcable_rules_copy.pop(str(rule))
        new_sorted = sorted(enforcable_rules_copy.keys())

        shield.remove_pos_rule(str(rule))
        shield.enforceable_rules.pop(str(rule))
        rule_chooser.set_rules_list(rule_chooser_rules_copy)
        rule_chooser.sorted_enforceable_rules = new_sorted

        log.info(f"Removed rule: {rule}")

        rules_snapshot = RuleSnapshot(
            enforceable_rules = shield.enforceable_rules,
            cancelable_rules = cancelable_rules_copy,
        )
        return rules_snapshot


def add_rule(rule, shield, rule_chooser, rule_str = None):
    if rule_str is not None:
        rule = string_to_rule(rule_str, False)
    if rule.polarity == False:
        cancelable_rule = dict()
        cancelable_rule[str(rule)] = [rule]

        cancelable_rules_copy = shield.cancelable_rules.copy()
        cancelable_rules_copy.update(cancelable_rule)

        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.append(len(rule_chooser.rules_list))

        new_sorted = sorted(cancelable_rules_copy.keys())

        shield.add_neg_rule(str(rule))
        shield.cancelable_rules = cancelable_rules_copy
        rule_chooser.set_rules_list(rule_chooser_rules_copy)
        rule_chooser.sorted_cancel_rules = new_sorted

        log.info(f"Added rule: {rule}")

        rules_snapshot = RuleSnapshot(
            enforceable_rules = shield.enforceable_rules,
            cancelable_rules = cancelable_rules_copy,
        )
        return rules_snapshot
    
    else:
        enforcable_rule = dict()
        enforcable_rule[str(rule)] = [rule]

        enforcable_rules_copy = shield.enforceable_rules.copy()
        enforcable_rules_copy.update(enforcable_rule)

        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.append(len(rule_chooser.rules_list))

        new_sorted = sorted(enforcable_rules_copy.keys())

        shield.add_pos_rule(str(rule))
        shield.enforceable_rules = enforcable_rules_copy
        rule_chooser.set_rules_list(rule_chooser_rules_copy)
        rule_chooser.sorted_enforceable_rules = new_sorted

        log.info(f"Added rule: {rule}")

        rules_snapshot = RuleSnapshot(
            enforceable_rules = shield.enforceable_rules,
            cancelable_rules = cancelable_rules_copy,
        )
        return rules_snapshot


def add_neg_rule(obs_rules, action, shield, rule_chooser, exclude_features):
    features_values = obs_rules.copy()
    features_values = shield.discretize(features_values).astype(int)
    relevant_facts = [
        f"f{fi}({value})"
        for fi, value in zip(shield.feature_indices, features_values)
        if fi not in exclude_features
    ]
    neg_rule = f"-action({action}) :- "
    for f in relevant_facts:
        neg_rule += f"{f},"
    neg_rule = neg_rule[:-1] + "."
    return add_rule(None, shield, rule_chooser, rule_str=neg_rule)     



def get_action(model, obs, obs_rules, shield, state, last_action, feature_extractor, rule_chooser, rules_snapshot, algo_name, act_logits, action_tensor):
    action, _states = model.predict(obs)
    triggers,triggered_actions, triggered_rules = shield.does_rule_trigger(obs_rules,rule_chooser,rules_snapshot)

    # print("triggered rules:")
    # if triggered_rules != None:
    #     (pos, neg) = triggered_rules
    #     for r in neg + pos:
    #         print(r[0])
    
    #print("orig. action :", action)
    if triggers:
        (pos_actions_triggered, neg_actions_triggered) = triggered_actions
        changed_action, activated_created_rules = change_action(action,pos_actions_triggered,neg_actions_triggered,algo_name,act_logits, action_tensor,
                                        triggered_rules, change_type="favor_enforce")
        if changed_action is not None:
            if len(activated_created_rules) > 0:
                log.info(f"Updated rule(s) used:\n {[str(r[0]) for r in activated_created_rules]}\nin state\n{state}")
            action = changed_action
    #print("changed action :", action, "\n")

    return action, triggered_rules


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