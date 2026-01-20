import hydra
from omegaconf import DictConfig
from util import setup_model, get_shield_number, get_feature_extractor, order_feature_indices
from rule_util import *
from legible.evaluate_policy import setup_shield
from legible.feature_and_rule_learn import get_features_and_failure_indication
from gym_pacman_rules.envs.featureExtractors import features_dict_to_array
from pacman_helper_asp import PacmanViolationClingoHelper
import sys, logging
import check_norms
import copy
from collections import deque 

log = logging.getLogger(__name__)

def setup(cfg):
    env, model, model_name, action_tensor = setup_model(cfg)
    shield_number = get_shield_number(cfg)

    #TODO what is difference between uncorr and improved?
    shield = setup_shield(cfg.env.name, cfg.env.level, cfg.training.steps_initial, cfg.rules.nr_features, False, False, exact_model_number=shield_number,
                           steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn", horizon=cfg.asp.horizon, radius=cfg.asp.radius)

    if shield is None:
        sys.exit("Could not load shield, check if it exists.")

    order_feature_indices(shield, env, cfg.rules.feature_extractor)
    rule_chooser = set_rules(shield)
    assert hasattr(shield, 'enforceable_rules')
    assert hasattr(shield, 'cancelable_rules')

    return env, model, action_tensor, shield, rule_chooser


def update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg):
    total_violations = 0
    last_n_violations = deque(maxlen=cfg.asp.horizon)
    last_n_states = deque(maxlen=cfg.asp.horizon)
    last_n_actions = deque(maxlen=cfg.asp.horizon)
    prev_env_states = deque(maxlen=cfg.asp.horizon)
    last_n_triggered_rules = deque(maxlen=cfg.asp.horizon)


    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    last_n_actions.append(None)
    last_n_violations.append(0)
    last_n_states.append(copy.deepcopy(env.unwrapped.game.state))
    prev_env_states.append(env.unwrapped.save_state())
    last_n_triggered_rules.append([])


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
                rules_snapshot = all_actions_blocked(obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_violations, prev_env_states, env, feature_extractor, model, action_tensor, cfg)

            action, triggered_rules = get_action(model, obs, obs_rules, shield, last_n_states[-1], rule_chooser, rules_snapshot, cfg.training.algorithm, act_logits, action_tensor)
            last_n_triggered_rules.append(triggered_rules)

            prev_eaten = env.unwrapped.has_eaten_ghost
            obs, reward, term, trunc, info = env.step(action)
            last_n_states.append(copy.deepcopy(env.unwrapped.game.state))
            last_n_actions.append(action)
            last_n_violations.append(check_norms.num_violations_detected(cfg.norms, last_n_states[-1], prev_eaten))
            prev_env_states.append(env.unwrapped.save_state())

            obs_t, vectorized_env = policy.obs_to_tensor(obs)
            obs_t = obs_t.to(action_tensor.device)

            if last_n_violations[-1] > 0:
                less_violations_possible = []
                for j, state in enumerate(last_n_states):
                    if j < cfg.asp.horizon - 1: 
                        log.info(
                            f"{cfg.asp.horizon - j - 1} step(s) before violation:\n{state}\n"
                            f"triggered rules:\n\t"
                            f"{'\n\t'.join(f'{r[0]}' for rs in last_n_triggered_rules[j+1] for r in rs)}\n"
                            f"action: {last_n_actions[j+1]}"
                        )
                        #TODO sum only over part of violations?
                        less_violations_possible.append(asp_helper.less_violations_possible(state, sum(last_n_violations), cfg.asp.horizon-j, prev_env_states[j]["has_eaten_ghost"]))
                    else:
                        log.info(
                            f"violation:\n{state}\n"
                        )
    
                if True in less_violations_possible:
                    prev_obs =  features_dict_to_array(feature_extractor.getFeatures(state=last_n_states[-2], action=last_n_actions[-1]))
                    rules_snapshot = add_neg_rule(prev_obs, last_n_actions[-1], shield, rule_chooser, cfg.rules.updates.exclude_features_in_neg_rules)
                    pos_triggered_rules = [r[0][0] for r in last_n_triggered_rules[-1] if r != [] and r[0][0].polarity]
                    categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(cfg.env.name, cfg.env.level, cfg.rules.feature_extractor)
                    for r in pos_triggered_rules:
                        rules_snapshot = remove_rule(r, shield, rule_chooser)
                        rules_snapshot = add_differing_enumerable_features(r, feature_extractor.getFeatures(last_n_states[-2],last_n_actions[-1]), categorical_features, shield,rule_chooser,cfg,model, env,action_tensor,feature_extractor)
                    backtrack(last_n_actions, last_n_states, prev_env_states, env, last_n_violations, feature_extractor, violation=False)

                    
                else:
                    log.info("Nothing to update, the number of violations cannot be decreased reliably.")

            if term or trunc:
                obs, info = env.reset()
                obs_t, vectorized_env = policy.obs_to_tensor(obs)
                obs_t = obs_t.to(action_tensor.device)
                last_n_states = deque(maxlen=cfg.asp.horizon)
                last_n_actions = deque(maxlen=cfg.asp.horizon)
                last_n_violations = deque(maxlen=cfg.asp.horizon)
                last_n_actions.append(None)
                last_n_violations.append(0)
                last_n_states.append(env.unwrapped.game.state)
                last_n_triggered_rules.append([])
                break
        
        if i % cfg.rules.updates.prune_interval == 0 and i != 0:
            log.info("Pruning rules.")
            rules_snapshot = prune_rule_set(model, env, action_tensor, feature_extractor, shield, rule_chooser, cfg)

    log.info(f"Total Violations: {total_violations}")


# TODO check if violations are handled correctly
def backtrack(last_n_actions, last_n_states, prev_env_states, env, last_n_violations, feature_extractor, violation=True):
    env.unwrapped.load_state(prev_env_states[-2])   
    prev_env_states.pop()
    if violation:
        last_n_violations.pop()
    last_n_states.pop()
    last_n_actions.pop()



def all_actions_blocked(obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_violations, prev_env_states, env, feature_extractor, model, action_tensor, cfg):
    triggers,triggered_actions, triggered_rules = shield.does_rule_trigger(obs_rules,rule_chooser,rules_snapshot)
    (pos_rules_triggered, neg_rules_triggered) = triggered_rules
    neg_actions = [r[0].rule_head.action for r in neg_rules_triggered]
    blocked_actions = set(neg_actions)
    actions_blocked_by_created_rules = {a: True for a in blocked_actions}

    for r in neg_rules_triggered:
        if r[0].mined == True:
            actions_blocked_by_created_rules[r[0].rule_head.action] = False

    if False not in list(actions_blocked_by_created_rules.values()):
        prev_obs =  features_dict_to_array(feature_extractor.getFeatures(state=last_n_states[-2], action=last_n_actions[-1]))
        backtrack(last_n_actions, last_n_states, prev_env_states, env, last_n_violations, feature_extractor, violation=False)
        rules_snapshot = add_neg_rule(prev_obs, last_n_actions[-1], shield, rule_chooser, cfg.rules.updates.exclude_features_in_neg_rules)
        return rules_snapshot
    else:
        log.info(f"All actions are blocked, adapting a mined rule")
        mined_neg_rule = max(list(filter(lambda r: r[0].mined == True, neg_rules_triggered)), key=lambda r: len(r[0].rule_body.conditions))[0]
        state_features = feature_extractor.getFeatures(last_n_states[-1],last_n_actions[-1])
        categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(cfg.env.name, cfg.env.level, cfg.rules.feature_extractor)
        rules_snapshot = remove_rule(mined_neg_rule, shield, rule_chooser)
        rules_snapshot = add_differing_enumerable_features(mined_neg_rule, state_features, categorical_features, shield, rule_chooser, cfg, model, env, action_tensor, feature_extractor)
        return rules_snapshot


def add_differing_enumerable_features(mined_rule,state_features, categorical_features, shield, rule_chooser, cfg, model, env, action_tensor, feature_extractor):
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
        if fi in cfg.rules.updates.exclude_features_in_neg_rules or fi in feature_indices_in_rule:
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
    log.info("Testing rules for retention")
    rules_snapshot = add_retaining_rules(adapted_rules, shield, rule_chooser, model, env, action_tensor, feature_extractor, cfg)
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