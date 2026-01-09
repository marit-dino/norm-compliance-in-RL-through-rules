import hydra
from omegaconf import DictConfig
from util import get_model_number, get_shield_number, get_feature_extractor
from rule_util import set_rules, save_rule_set, RuleSnapshot, get_action, remove_rule, add_rule, prune_rule_set
from legible.env_util import create_environment_and_modelname_for_oftendeeprl
from legible.evaluate_policy import setup_shield
from legible.create_rules_pacman import string_to_rule
from legible.rule_learning.util import load_model
from legible.feature_and_rule_learn import get_features_and_failure_indication
from gym_pacman_rules.envs.featureExtractors import features_dict_to_array
from pacman_helper_asp import PacmanViolationClingoHelper
import sys, logging, torch
import check_norms
import copy
from collections import deque 

log = logging.getLogger(__name__)

def setup(cfg):
    norm_descriptor = f"{'_'.join(cfg.norms)}"
    config_str = f"{norm_descriptor}__{str(cfg.asp.horizon)}_{str(cfg.asp.radius)}"

    env, model_name, model_path = create_environment_and_modelname_for_oftendeeprl("norm_guided_dqn", cfg.env.name,
                                                                                    cfg.env.level, config_str, 
                                                                                    cfg.training.steps_initial, cfg.training.steps_norm,
                                                                                    cfg.training.feature_extractor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = env.action_space.n
    action_tensor = torch.tensor(range(num_actions), device=device)

    model_number = get_model_number(cfg)
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

    feature_to_index = {
        name: idx for idx, name in enumerate(features)
    }
    ordered_feature_indices = [feature_to_index[fv[0]] for fv in ordered_features]

    shield.feature_indices = ordered_feature_indices
    shield, rule_chooser = set_rules(shield)
    assert hasattr(shield, 'enforceable_rules')
    assert hasattr(shield, 'cancelable_rules')

    return env, model, action_tensor, shield, rule_chooser


def update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg):
    total_violations = 0
    last_n_violations = deque(maxlen=cfg.asp.horizon+1)
    last_n_states = deque(maxlen=cfg.asp.horizon+1)
    last_n_actions = deque(maxlen=cfg.asp.horizon+1)
    prev_env_states = deque(maxlen=cfg.asp.horizon+1)

    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    last_n_actions.append(None)
    last_n_violations.append(0)
    last_n_states.append(copy.deepcopy(env.unwrapped.game.state))
    prev_env_states.append(env.unwrapped.save_state())

    feature_extractor = get_feature_extractor(cfg.rules.feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)
    asp_helper = PacmanViolationClingoHelper(cfg.asp.horizon, cfg.asp.radius, number_of_ghosts(cfg.env.level), cfg.norms, num_norms=len(cfg.norms))

    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )

    for i in range(cfg.rules.updates.episodes):
        log.info(f"Episode {i+1}/{cfg.rules.updates.episodes}")

        while True:
            total_violations += last_n_violations[-1]
            action, _states = model.predict(obs)
            q_values = policy.q_net(obs_t).squeeze()
            act_logits = q_values        
            obs_rules = features_dict_to_array(feature_extractor.getFeatures(last_n_states[-1],action))

            if len(shield.get_blocked_actions(obs_rules)) == 4:
                rules_snapshot = all_actions_blocked(obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_violations, prev_env_states, env, feature_extractor, cfg)
                rules_snapshot = prune_rule_set(model, env, action_tensor, feature_extractor, shield, rule_chooser, cfg)
     
            action, triggered_rules = get_action(model, obs, obs_rules, shield, last_n_states[-1], rule_chooser, rules_snapshot, cfg.training.algorithm, act_logits, action_tensor)
            
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
                    log.info(f"{cfg.asp.horizon - j} step(s) before violation: \n{state}")
                    if j < cfg.asp.horizon:
                        less_violations_possible.append(asp_helper.less_violations_possible(last_n_states_copy.popleft(), sum(last_n_violations), cfg.asp.horizon-j+1))
                if True in less_violations_possible:
                    prev_state, prev_action, prev_obs = backtrack(last_n_actions, last_n_states, prev_env_states, env, last_n_violations, feature_extractor, violation=False)
                    rules_snapshot = add_neg_rule(prev_obs, prev_action, shield, rule_chooser, cfg.rules.updates.exclude_features_in_neg_rules)
                    continue
                else:
                    log.info("Nothing to update, the number of violations cannot be decreased reliably.")

            if term or trunc:
                obs, info = env.reset()
                obs_t, vectorized_env = policy.obs_to_tensor(obs)
                obs_t = obs_t.to(action_tensor.device)
                last_n_states = deque(maxlen=cfg.asp.horizon+1)
                last_n_actions = deque(maxlen=cfg.asp.horizon+1)
                last_n_violations = deque(maxlen=cfg.asp.horizon+1)
                last_n_actions.append(None)
                last_n_violations.append(0)
                last_n_states.append(env.unwrapped.game.state)
                break
        
        # collect_data()
        # prune_rule_set()
        # optimize_rule_set()

    log.info(f"Total Violations: {total_violations}")


def backtrack(last_n_actions, last_n_states, prev_env_states, env, last_n_violations, feature_extractor, violation=True):
    env.unwrapped.load_state(prev_env_states[-2])   
    prev_env_states.pop()
    if violation:
        last_n_violations.pop()
    prev_obs =  features_dict_to_array(feature_extractor.getFeatures(state=last_n_states[-2], action=last_n_actions[-1]))
    return last_n_states.pop(), last_n_actions.pop(), prev_obs



def all_actions_blocked(obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_violations, prev_env_states, env, feature_extractor, cfg):
    triggers,triggered_actions, triggered_rules = shield.does_rule_trigger(obs_rules,rule_chooser,rules_snapshot)
    (pos_rules_triggered, neg_rules_triggered) = triggered_rules
    neg_actions = [r[0].rule_head.action for r in neg_rules_triggered]
    blocked_actions = set(neg_actions)
    actions_blocked_by_created_rules = {a: True for a in blocked_actions}

    for r in neg_rules_triggered:
        if r[0].mined == True:
            actions_blocked_by_created_rules[r[0].rule_head.action] = False

    if False not in list(actions_blocked_by_created_rules.values()):
        log.info(f"Backtracking, all actions are blocked by created rules")
        prev_state, prev_action, prev_obs = backtrack(last_n_actions, last_n_states, prev_env_states, env, last_n_violations, feature_extractor, violation=False)
        rules_snapshot = add_neg_rule(prev_obs, last_n_actions[-1], shield, rule_chooser, cfg.rules.updates.exclude_features_in_neg_rules)
        return rules_snapshot
    else:
        log.info(f"All actions are blocked, adapting a mined rule")
        mined_neg_rule = max(list(filter(lambda r: r[0].mined == True, neg_rules_triggered)), key=lambda r: len(r[0].rule_body.conditions))[0]
        state_features = feature_extractor.getFeatures(last_n_states[-1],last_n_actions[-1])
        categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(cfg.env.name, cfg.env.level, cfg.rules.feature_extractor)
        rules_snapshot = add_differing_enumerable_features(mined_neg_rule, state_features, categorical_features, shield, rule_chooser, cfg.rules.updates.exclude_features_in_neg_rules)
        return rules_snapshot


def add_differing_enumerable_features(mined_rule,state_features, categorical_features, shield, rule_chooser, exclude_features_in_neg_rules):
    discretized_features = shield.discretize(features_dict_to_array(state_features)).astype(int)

    feature_facts = {
        fi: value
        for fi, value in zip(shield.feature_indices,discretized_features)
    }
    rule_copy = copy.deepcopy(mined_rule)
    rule_copy.mined = False
    feature_indices_in_rule = [f.feature for f in mined_rule.rule_body.conditions]
    adapted_rules = [rule_copy]
    for fi in feature_facts.keys():
        if fi in exclude_features_in_neg_rules or fi in feature_indices_in_rule:
            continue
        elif fi in categorical_features:
            for i,r in enumerate(adapted_rules):
                adapted_rules[i] = r.add_feature(fi, abs(feature_facts[fi]-1))
        else:
            tmp_rule_list = []
            for i,r in enumerate(adapted_rules):
                for interval,(l,u) in enumerate(shield.feature_intervals[fi]):
                    if feature_facts[fi] != interval:
                        tmp_rule_list.append(r.add_feature(fi, interval))
            adapted_rules = tmp_rule_list
    rules_snapshot = remove_rule(mined_rule, shield, rule_chooser)
    for r in adapted_rules:
        rules_snapshot = add_rule(r, shield, rule_chooser)
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


def check_retention(rule, shield, rule_chooser):
    if rule.polarity == False:
        polarity_rule_set = rule_chooser.sorted_cancel_rules
    else:
        polarity_rule_set = rule_chooser.sorted_enforce_rules

        
    # check combination
    # always check for one feature diff and then recursive call?
    combine_rules(rule, shield, rule_chooser)

    return True


def combine_rules(rule, polarity_rule_set, shield, rule_chooser):
    one_feature_diff = [r for r in polarity_rule_set if len(set(rule.rule_body.conditions) - set(r.rule_body.conditions)) == 1]
    # TODO

def number_of_ghosts(level):
    if level.startswith("small"):
        return 2
    else:
        return 4


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    env, model, action_tensor, shield, rule_chooser = setup(cfg)
    update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg)
    save_rule_set(shield, cfg)


if __name__ == "__main__":
    main()