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
import copy, random
import itertools
from collections import deque 

log = logging.getLogger(__name__)

def setup(cfg):
    """ Loads the environment and model, as well as setups the shield and rule chooser.

    Args:
        cfg (DictConfig): config object provided by hydra containing all parameters

    Returns:
        PacmanEnv, model, Tensor, AspShield, RuleChooser: environment, model, action tensor, shield and rule chooser
    """
    env, model, model_name, action_tensor = setup_model(cfg)
    shield_number = get_shield_number(cfg)
    config_str = f"{cfg.norm.id}__{str(cfg.asp.horizon)}_{str(cfg.asp.radius)}"

    shield = setup_shield(cfg.env.name, cfg.level.id, cfg.training.steps_initial, cfg.level.rules.nr_features, False, False, exact_model_number=shield_number,
                           steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn", config_str=config_str)

    if shield is None:
        sys.exit("Could not load shield, check if it exists.")

    order_feature_indices(shield, env, cfg.rules.feature_extractor)
    rule_chooser = set_rules(shield)
    assert hasattr(shield, 'enforceable_rules')
    assert hasattr(shield, 'cancelable_rules')

    return env, model, action_tensor, shield, rule_chooser

def update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg):
    """ Updates the rule set by removing and adding rules in order to reduce the number of violations.
        The shield and action tensor are modified to contain this updated rule set.

    Args:
        env (PacmanEnv): environment
        model : model
        action_tensor (Tensor): tensor containing the available actions
        shield (AspShield): shield that is set up with the initially mined rules
        rule_chooser (RuleChooser): rule chooser that is also set up with the initially mined rules
        cfg (DictConfig): config object provided by hydra containing all parameters
    """
    last_n_violations = deque(maxlen=cfg.asp.horizon)
    last_n_obs_rules = deque(maxlen=cfg.asp.horizon)
    last_n_states = deque(maxlen=cfg.asp.horizon)
    last_n_actions = deque(maxlen=cfg.asp.horizon)
    prev_env_states = deque(maxlen=cfg.asp.horizon)
    last_n_triggered_rules = deque(maxlen=cfg.asp.horizon)
    feature_extractor = get_feature_extractor(cfg.rules.feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)


    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    last_n_actions.append(None)
    last_n_violations.append(0)
    last_n_obs_rules.append(features_dict_to_array(feature_extractor.getFeatures(env.unwrapped.game.state,None,env.unwrapped.eaten_ghost)))
    last_n_states.append(env.unwrapped.game.state)
    prev_env_states.append(env.unwrapped.save_state())
    last_n_triggered_rules.append([])


    asp_helper = PacmanViolationClingoHelper(cfg.asp.horizon, cfg.asp.radius, number_of_ghosts(cfg.level.id), cfg.norm.id)

    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )

    rule_set_changes_over_period = 0
    num_backtracks = 0
    last_backtrack = -1

    for i in range(cfg.rules.updates.episodes):
        log.info(f"Episode {i+1}/{cfg.rules.updates.episodes}")

        while True:
            q_values = policy.q_net(obs_t).squeeze()
            act_logits = q_values    

            while len(shield.get_blocked_actions(last_n_obs_rules[-1], rules_snapshot)) == 4: 
                rule_set_changes_over_period += 1
                print(num_backtracks)
                rules_snapshot, backtrack_updated_rules = all_actions_blocked(last_n_obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_violations, last_n_triggered_rules, prev_env_states, env, feature_extractor, model, action_tensor, cfg)
                if backtrack_updated_rules:
                    num_backtracks += 1
                    last_backtrack = i
                if num_backtracks > cfg.asp.horizon * 4:
                    break

            if i - last_backtrack > cfg.asp.horizon:
                num_backtracks = 0
                last_backtrack = -1
            
            action, triggered_rules = get_action(model, obs, last_n_obs_rules[-1], shield, env.unwrapped.game.state, rule_chooser, rules_snapshot, cfg.training.algorithm, act_logits, action_tensor)
            last_n_triggered_rules.append(triggered_rules)

            prev_eaten = env.unwrapped.eaten_ghost
            obs, reward, term, trunc, info = env.step(action)

            moved_north = info["moved_north"]

            last_n_obs_rules.append(features_dict_to_array(feature_extractor.getFeatures(env.unwrapped.game.state,action,env.unwrapped.eaten_ghost)))
            last_n_actions.append(action)
            last_n_states.append(env.unwrapped.game.state)
            last_n_violations.append(check_norms.num_violations_detected(cfg.norm.id, last_n_states[-1], prev_eaten, moved_north))
            prev_env_states.append(env.unwrapped.save_state())

            obs_t, vectorized_env = policy.obs_to_tensor(obs)
            obs_t = obs_t.to(action_tensor.device)

            if last_n_violations[-1] > 0:
                less_violations_possible = []
              
                for j, state in enumerate(last_n_states):
                    if j < len(last_n_states) - 1: 
                        log.info(
                            f"{len(last_n_states) - j - 1} step(s) before violation:\n{state}\n"
                            f"triggered rules:\n\t"
                            f"{'\n\t'.join(f'{r[0]}' for rs in last_n_triggered_rules[j+1] for r in rs)}\n"
                            f"action: {last_n_actions[j+1]}\n"
                            f"Number of violations from this to last state: {sum(itertools.islice(last_n_violations, j+1, len(last_n_violations)))}"
                        )
                        less_violations_possible.append(asp_helper.less_violations_possible(state, sum(itertools.islice(last_n_violations, j+1, len(last_n_violations))), cfg.asp.horizon-j, prev_env_states[j]["eaten_ghost"]))
                    else:
                        log.info(
                            f"violation:\n{state}\n"
                        )
                if True in less_violations_possible:
                    rule_set_changes_over_period += 1
                    index = len(less_violations_possible) - 1 - less_violations_possible[::-1].index(True)
                    rules_snapshot = add_neg_rule(last_n_obs_rules[index], last_n_actions[index+1], shield, rule_chooser, get_excluded_features(cfg))
                    pos_triggered_rules = [r[0][0] for r in last_n_triggered_rules[index+1] if r != [] and r[0][0].polarity]
                    categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(cfg.env.name, cfg.level.id, cfg.rules.feature_extractor)
                    for r in pos_triggered_rules:
                        rules_snapshot = remove_rule(r, shield, rule_chooser)
                        if cfg.rules.updates.add_rule_variations:
                            rules_snapshot = add_differing_enumerable_features(r, last_n_obs_rules[index], categorical_features, shield,rule_chooser,cfg,model, env,action_tensor,feature_extractor)
                    backtrack(len(last_n_states) - index - 1, last_n_actions, last_n_states, prev_env_states, env, last_n_violations, last_n_triggered_rules, last_n_obs_rules)

                else:
                    log.info("Nothing to update, the number of violations cannot be decreased reliably.")

            if term or trunc:
                obs, info = env.reset()
                obs_t, vectorized_env = policy.obs_to_tensor(obs)
                obs_t = obs_t.to(action_tensor.device)
                last_n_obs_rules = deque(maxlen=cfg.asp.horizon)
                last_n_states = deque(maxlen=cfg.asp.horizon)
                last_n_actions = deque(maxlen=cfg.asp.horizon)
                last_n_violations = deque(maxlen=cfg.asp.horizon)
                last_n_triggered_rules = deque(maxlen=cfg.asp.horizon)
                last_n_actions.append(None)
                last_n_violations.append(0)
                last_n_obs_rules.append(features_dict_to_array(feature_extractor.getFeatures(env.unwrapped.game.state,None,env.unwrapped.eaten_ghost)))
                last_n_states.append(env.unwrapped.game.state)
                prev_env_states = deque(maxlen=cfg.asp.horizon)
                prev_env_states.append(env.unwrapped.save_state())
                last_n_triggered_rules.append([])
                num_backtracks = 0
                last_backtrack = -1
                break
        
        if i % cfg.rules.updates.prune_interval == 0 and i != 0:
            if rule_set_changes_over_period > 1 and cfg.rules.updates.prune_rules:
                log.info("Pruning rules.")
                rules_snapshot = prune_rule_set(model, env, action_tensor, feature_extractor, shield, rule_chooser, cfg)
            rule_set_changes_over_period = 0


def backtrack(steps, last_n_actions, last_n_states, prev_env_states, env, last_n_violations, last_n_triggered_rules, last_n_obs_rules):
    """ Backtracks a specified number of steps and updates those lists and variables that are affected by this.

    Args:
        steps (int): steps to backtrack
        last_n_actions (deque[int]): deque containing the last n taken actions
        last_n_states (deque[GameState]): deque containing the last n visited states
        prev_env_states (deque[dict]): deque containing representations of the last n environment configurations
        env (PacmanEnv): environment
        last_n_violations (deque[int]): deque containing the number of violations over the last n steps
        last_n_triggered_rules (deque[list[Rule]]): deque containing the rules that triggered over the last n steps
        last_n_obs_rules (deque[NDArray]): deque containing the last n observations processed by the feature extractor used for the rules and features_dict_to_array()
    """
    log.info(f"backtracking {steps} step(s) to:") 
    for i in range(steps):
        prev_env_states.pop()
        last_n_violations.pop()
        last_n_states.pop()
        last_n_actions.pop()
        last_n_triggered_rules.pop()
        last_n_obs_rules.pop()
    env.unwrapped.load_state(prev_env_states[-1])  
    log.info(f"\n{env.unwrapped.game.state}")


def all_actions_blocked(last_n_obs_rules, shield, rule_chooser, rules_snapshot, last_n_actions, last_n_states, last_n_violations,last_n_triggered_rules, prev_env_states, env, feature_extractor, model, action_tensor, cfg):
    """ This method either backtracks or adapts/removes a rule in case all possible actions are currently blocked by rules.
        If one of the blocking rules was mined, it is adapted/removed to allow the agent to take this action.
        If none of the blocking rules was mined, we backtrack to the previous state and add a rule which prevents the agent from ending up in this state again.

    Args:
        last_n_obs_rules (deque[NDArray]): deque containing the last n observations processed by the feature extractor used for the rules and features_dict_to_array()
        shield (AspShield): shield containing the current rule set
        rule_chooser (RuleChooser): rule chooser containing the current rule set
        rules_snapshot (RuleSnapshot): snapshot of the current state of the rule set
        last_n_actions (deque[int]): deque of the last n actions taken
        last_n_states (deque[GameState]): deque of the last n visited states
        last_n_violations (deque[int]): deque of number of violations per state over the last n states
        last_n_triggered_rules (deque[list[Rule]]): deque containing the triggered rules over the last n steps
        prev_env_states (deque[dict]): deque containing representations of the last n environment configurations
        env (PacmanEnv): environment
        feature_extractor (FeatureExtractor): feature extractor used for the rules
        model : model
        action_tensor (Tensor): tensor containing the available actions
        cfg (DictConfig): config object provided by hydra containing all parameters

    Returns:
        RuleSnapshot, bool: snapshot of the new rule set, flag signaling that backtracking due to actions being blocked by only updated rules happened
    """
    triggers,triggered_actions, triggered_rules = shield.does_rule_trigger(last_n_obs_rules[-1],rule_chooser,rules_snapshot)
    (pos_rules_triggered, neg_rules_triggered) = triggered_rules
    neg_actions = [r[0].rule_head.action for r in neg_rules_triggered]
    blocked_actions = set(neg_actions)
    actions_blocked_by_created_rules = {a: True for a in blocked_actions}

    for r in neg_rules_triggered:
        if r[0].mined == True:
            actions_blocked_by_created_rules[r[0].rule_head.action] = False

    if False not in list(actions_blocked_by_created_rules.values()):
        if len(last_n_obs_rules) > 1:
            log.info(f"All actions are blocked by updated rules, backtracking one step.")
            prev_obs = last_n_obs_rules[-2]
            backtrack(1, last_n_actions, last_n_states, prev_env_states, env, last_n_violations, last_n_triggered_rules, last_n_obs_rules)
            rules_snapshot = add_neg_rule(prev_obs, last_n_actions[-1], shield, rule_chooser, get_excluded_features(cfg))
            return rules_snapshot, True
        else:
            log.info(f"All actions are blocked by updated rules and not possible to go back, removing rule(s).")
            action = random.randint(0,3)
            blocking_rules = [r[0] for r in neg_rules_triggered if r[0].rule_head.action == action]
            for r in blocking_rules:
                rules_snapshot =  remove_rule(r, shield, rule_chooser)
            return rules_snapshot, False
    else:
        log.info(f"All actions are blocked, adapting a mined rule")
        mined_neg_rule = max(list(filter(lambda r: r[0].mined == True, neg_rules_triggered)), key=lambda r: len(r[0].rule_body.conditions))[0]
        state_features = last_n_obs_rules[-1]
        categorical_features, failure_indicator, nr_features_all, sample_reconstruction, groups_of_similar = get_features_and_failure_indication(cfg.env.name, cfg.level.id, cfg.rules.feature_extractor)
        rules_snapshot = remove_rule(mined_neg_rule, shield, rule_chooser)
        if cfg.rules.updates.add_rule_variations:
            rules_snapshot = add_differing_enumerable_features(mined_neg_rule, state_features, categorical_features, shield, rule_chooser, cfg, model, env, action_tensor, feature_extractor)
        return rules_snapshot, False


def add_differing_enumerable_features(mined_rule,obs_rules, categorical_features, shield, rule_chooser, cfg, model, env, action_tensor, feature_extractor):
    """ Takes a rule and an observation and makes the rule more specific (may also result in several rules) in order to exclude the application of rule in case of the specified observation.

    Args:
        mined_rule (Rule): rule which is to be adapted
        obs_rules (NDArray): observation processed by the feature extractor used for the rules and features_dict_to_array(), this is the observation that will be excluded from the rule body
        categorical_features (list[int]): list containing the number of those features which are categorical
        shield (AspShield): shield
        rule_chooser (RuleChooser): rule chooser
        cfg (DictConfig): config object provided by hydra containing all parameters
        model : model
        env (PacmanEnv): environment
        action_tensor (Tensor): tensor containing the available actions
        feature_extractor (FeatureExtractor): feature extractor used for the rules

    Returns:
        RuleSnapshot: snapshot of the new rule set
    """
    discretized_features = shield.discretize(obs_rules).astype(int)

    feature_facts = {
        fi: value
        for fi, value in zip(shield.feature_indices,discretized_features)
    }
    rule_copy = copy.deepcopy(mined_rule)
    rule_copy.mined = False
    feature_indices_in_rule = [f.feature for f in mined_rule.rule_body.conditions]

    feature_subset = list(set(range(0, cfg.level.rules.nr_features)) - set(feature_indices_in_rule) - set(get_excluded_features(cfg)))

    adapted_rules = [rule_copy]
    for fi in feature_facts.keys():
        if fi not in feature_subset:
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

    adapted_rules_selection = random.sample(adapted_rules, min(len(adapted_rules), 100))
    log.info("Testing rules for retention")
    snapshot = env.unwrapped.save_state()
    rules_snapshot = add_retaining_rules(adapted_rules_selection, shield, rule_chooser, model, env, action_tensor, feature_extractor, cfg)
    env.unwrapped.load_state(snapshot)
    return rules_snapshot
       

def add_neg_rule(obs_rules, action, shield, rule_chooser, exclude_features):
    """ Adds a negative rule to the rule set, where the body is specified by the given observation.

    Args:
        obs_rules (NDArray): observation processed by the feature extractor used for the rules and features_dict_to_array(), this is the observation that will form the body of the rule
        action (int): action that will be in the head of the rule
        shield (AspShield): shield to which the rule is added
        rule_chooser (RuleChooser): rule chooser to which the rule is added
        exclude_features (list[int]): features that are ignored when building the rule body

    Returns:
        RuleSnapshot: snapshot of the new rule set
    """
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
    """ Returns the number of ghosts in this environment.

    Args:
        level (string): level descriptor, e.g. smallClassic

    Returns:
        int: number of ghosts
    """
    if level.startswith("small") or level.startswith("medium"):
        return 2
    else:
        return 4

def get_excluded_features(cfg):
    if "original" in cfg.level.id:
        return cfg.norm.exclude_features_in_rules_orig
    else:
        return cfg.norm.exclude_features_in_rules


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    env, model, action_tensor, shield, rule_chooser = setup(cfg)
    update_rule_set(env, model, action_tensor, shield, rule_chooser, cfg)
    save_rule_set(shield, cfg)


if __name__ == "__main__":
    main()
