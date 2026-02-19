import logging
from gym_pacman_rules.envs.featureExtractors import features_dict_to_array
from legible.rule_learning.util import save_pickle
from legible.shield.shields import RuleChooser 
from legible.create_rules_pacman import string_to_rule
from tqdm import tqdm
import copy

log = logging.getLogger(__name__)



class RuleSnapshot:
    """ Used to keep the current "rule state" consistent among e.g. the shield and rule chooser.
    """
    enforceable_rules = dict()
    cancelable_rules = dict()
    rule_indices = []

    def __init__(self, enforceable_rules, cancelable_rules):
        self.enforceable_rules = enforceable_rules 
        self.cancelable_rules = cancelable_rules  
        self.rule_indices = range(0, len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))



def collect_data(model, env, action_tensor,feature_extractor, shield, rule_chooser,training_algorithm, episodes):
    """ Collects data on which rules triggered over several episodes.

    Args:
        model : model
        env (PacmanEnv): environment
        action_tensor (Tensor): tensor for the available actions
        feature_extractor (FeatureExtractor): feature extractor used for the rule mining step
        shield (AspShield): shield to check whether rules apply
        rule_chooser (RuleChooser): rule chooser for the rules mined in the previous step (+ added updated rules)
        training_algorithm (string): string to describe the algorithm used to train the initial policy (in practice, this will always be 'norm_guided_dqn')
        episodes (int): number of episodes to collect data over

    Returns:
        list[list[Rule]]: list of lists containing the triggered rules per step
    """
    triggered_rules_list = []

    obs, _info = env.reset()
    policy = model.policy
    obs_t, _vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)

    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )

    for i in tqdm(range(episodes)):
        while True:
            action, _states = model.predict(obs)
            q_values = policy.q_net(obs_t).squeeze()
            act_logits = q_values        
            obs_rules = features_dict_to_array(feature_extractor.getFeatures(env.unwrapped.game.state,action,env.unwrapped.eaten_ghost))
            
            action, triggered_rules = get_action(model, obs, obs_rules, shield, env.unwrapped.game.state,rule_chooser, rules_snapshot, training_algorithm, act_logits, action_tensor)
            triggered_rules_list.append(triggered_rules)

            obs, _reward, term, trunc, _info = env.step(action)
            obs_t, _vectorized_env = policy.obs_to_tensor(obs)
            obs_t = obs_t.to(action_tensor.device)

            if term or trunc:
                obs, _info = env.reset()
                obs_t, _vectorized_env = policy.obs_to_tensor(obs)
                obs_t = obs_t.to(action_tensor.device)
                break

    return triggered_rules_list
        

def prune_rule_set(model, env, action_tensor, feature_extractor, shield, rule_chooser, cfg):
    """Prunes the updated rules in the current rule set based on their usage and whether they can be merged.

    Args:
        model : model
        env (PacmanEnv): environment
        action_tensor (Tensor): tensor for the available actions
        feature_extractor (FeatureExtractor): feature extractor used for the rule mining step
        shield (AspShield): shield to check whether rules apply
        rule_chooser (RuleChooser): rule chooser for the rules mined in the previous step (+ added updated rules)
        cfg (DictConfig): config object provided by hydra containing all parameters

    Returns:
        RuleSnapshot: snapshot of the pruned rule set
    """
    rules = [r[0] for r in (list(shield.cancelable_rules.values()) + list(shield.enforceable_rules.values())) if r[0].mined == False]
    if rules == []:
        return RuleSnapshot(
            enforceable_rules = shield.enforceable_rules,
            cancelable_rules = shield.cancelable_rules,
        )

    logging.disable(logging.CRITICAL)
    rule_data = collect_data(model, env, action_tensor, feature_extractor, shield, rule_chooser, cfg.training.algorithm, cfg.rules.updates.data_collection_episodes)
    logging.disable(logging.NOTSET)
    rules_snapshot = remove_unused_rules(rule_data, rule_chooser, shield)
    
    rules = [r[0] for r in (list(shield.cancelable_rules.values()) + list(shield.enforceable_rules.values()))]
    for r in rules:
        rules_snapshot = merge_rules(r, shield, rule_chooser)
    return rules_snapshot


def merge_rules(rule, shield, rule_chooser):
    """Merges rules if they have the same features in the body, but each of them evaluates one feature (the same among all of these rules)
        to a different value. And these values cover all possible values that this feature can take on.

    Args:
        rule (Rule): rule for which to check whether it can be merged
        shield (AspShield)
        rule_chooser (RuleChooser)

    Returns:
        RuleSnapshot: snapshot of the new, potentially merged, rule set
    """
    rules = [r[0] for r in (list(shield.cancelable_rules.values()) + list(shield.enforceable_rules.values()))]
    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )
    one_differing_val_rules = [r for r in rules 
                                if len(list(set(rule.rule_body.conditions) - set(r.rule_body.conditions))) == 1 
                                and len(list(set(r.rule_body.conditions) - set(rule.rule_body.conditions))) == 1
                                and [f.feature for f in r.rule_body.conditions] == [f.feature for f in rule.rule_body.conditions]]
    if len(one_differing_val_rules) == 0:
        return rules_snapshot
    
    f = list(set(rule.rule_body.conditions) - set(one_differing_val_rules[0].rule_body.conditions))[0].feature
    v = list(set(rule.rule_body.conditions) - set(one_differing_val_rules[0].rule_body.conditions))[0].valuation
    one_differing_val_rules.append(rule)

    #make sure that all have different values for the feature
    if len(set(one_differing_val_rules)) == len(shield.feature_intervals[f]):
        log.info("Merging rules.")
        for r in one_differing_val_rules:
            rules_snapshot = remove_rule(r, shield, rule_chooser)
        merged_rule = rule.remove_feature(f, v)
        rules_snapshot = add_rule(merged_rule, shield, rule_chooser)
        return merge_rules(merged_rule, shield, rule_chooser)
    
    return rules_snapshot



def remove_unused_rules(data, rule_chooser, shield):
    """Removes those (non-mined) rules, which are not used (over a certain number of episodes)

    Args:
        data (list[list[Rule]]): list of lists of used rules per episode, as returned by collect_data
        rule_chooser (RuleChooser)
        shield (AspShield)

    Returns:
        RuleSnapshot: snapshot of the potentially decimated rule set
    """
    non_mined_rules = [r[0] for r in (list(shield.cancelable_rules.values()) + list(shield.enforceable_rules.values())) if not r[0].mined]

    flattened_data = [r[0] for rs in data for rt in rs for r in rt]
    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )

    for r in non_mined_rules:
        if r not in flattened_data:
            rules_snapshot = remove_rule(r, shield, rule_chooser)
    return rules_snapshot


def add_retaining_rules(rules, shield, rule_chooser, model, env, action_tensor, feature_extractor, cfg):
    """Adds those rules to the rule set which are used at least once in a certain number of episodes.

    Args:
        rules (list[Rule]): list of rules which may be added
        shield (AspShield)
        rule_chooser (RuleChooser)
        model 
        env (PacmanEnv): environment
        action_tensor (Tensor): tensor containing the available actions
        feature_extractor (FeatureExtractor): the same as used in the rule mining step
        cfg (DictConfig): config object provided by hydra containing all parameters

    Returns:
        RuleSnapshot: snapshot of the new rule set
    """
    shield_upd = copy.deepcopy(shield)
        
    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )

    rule_chunks = [rules[i:i + 25] for i in range(0, len(rules), 25)]
    flattened_data = []

    for rc in rule_chunks:
        shield_upd = copy.deepcopy(shield)
        for r in rc:
            if r.polarity:
                shield_upd.add_pos_rule(r)
            else:
                shield_upd.add_neg_rule(r)

        rule_chooser_upd = set_rules(shield_upd)

        logging.disable(logging.CRITICAL)
        rule_data = collect_data(model, env, action_tensor, feature_extractor, shield_upd, rule_chooser_upd, cfg.training.algorithm, cfg.rules.updates.data_collection_episodes)
        logging.disable(logging.NOTSET)
        flattened_data = flattened_data + ([r[0] for rs in rule_data for rt in rs for r in rt])


    for r in rules:
        if r in flattened_data:
            rules_snapshot = add_rule(r, shield, rule_chooser)
    return rules_snapshot


# adapted from legible
def set_rules(shield):
    """ Sets the enforceable and cancelable rules of the shield and initializes the corresponding rule chooser.

    Args:
        shield (AspShield): shield of which the enforceable and cancelable rules should be set

    Returns:
        RuleChooser: rule chooser corresponding to the shield
    """
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
    return rule_chooser
    

def save_rule_set(shield, cfg):
    """ Stores the current rule set (as stored in the shield) in a pickle file.

    Args:
        shield (AspShield): shield containing the rules to store
        cfg (DictConfig): config object provided by hydra containing all parameters
    """
    config_str = f"{cfg.norm.id}__{cfg.asp.horizon}_{cfg.asp.radius}"
    shield_name = f"pickles/shields/uncorr/"\
                    f"norm_guided_dqn__{config_str}__{cfg.env.name.replace('/', '_')}_{cfg.env.level}_feat_{cfg.rules.nr_features}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_shield_updated"

    if cfg.rules.shield_number is None:
        save_pickle(shield_name,shield)
    else:
        shield_name = f"{shield_name}_updated_{cfg.rules.shield_number}.pkl"
        save_pickle(shield_name,shield,exact_match=True)


def get_action(model, obs, obs_rules, shield, state, rule_chooser, rules_snapshot, algo_name, act_logits, action_tensor):
    """ Returns the action that should be taken based on the model and the rule set.
        It also returns the rules that triggered in this state.

    Args:
        model : model
        obs : observation as returned by env.step()
        obs_rules : observation that has been prepared for checking whether it triggers rules (i.e. using features_dict_to_array)
        shield (AspShield): shield containing the rules
        state : env.unwrapped.game.state
        rule_chooser (RuleChooser): rule chooser used to check whether rules trigger
        rules_snapshot (RuleSnapshot): current snapshot of the rule set
        algo_name (string): name of the algorithm used to train the initial policy (in practice this will always be 'norm_guided_dqn')
        act_logits (list[double]): action logits (output of model)
        action_tensor (Tensor): tensor containing the available actions

    Returns:
        int, list[Rule]: number of chosen action and rules that triggered
    """
    action, _state = model.predict(obs)
    triggers,triggered_actions, triggered_rules = shield.does_rule_trigger(obs_rules,rule_chooser,rules_snapshot)
    if triggered_rules == None:
        triggered_rules = []
    
    from legible.evaluate_policy import change_action

    if triggers:
        (pos_actions_triggered, neg_actions_triggered) = triggered_actions
        changed_action, activated_created_rules = change_action(action,pos_actions_triggered,neg_actions_triggered,algo_name,act_logits, action_tensor,
                                        triggered_rules, change_type="favor_enforce")
        if changed_action is not None:
            if len(activated_created_rules) > 0:
                log.info(f"Updated rule(s) used:\n\t{"\n\t".join([str(r[0]) for r in activated_created_rules])}\nin state\n{state}")
            action = changed_action
    
    return action, triggered_rules


def remove_rule(rule, shield, rule_chooser):
    """ Removes a rule from the shield and rule chooser

    Args:
        rule (Rule): rule to remove
        shield (AspShield): shield to remove the rule from
        rule_chooser (RuleChooser): rule chooser to remove the rule from

    Returns:
        RuleSnapshot: snapshot of the new rule set
    """
    if rule.polarity == False:
        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.pop()
        cancelable_rules_copy = shield.cancelable_rules.copy()
        cancelable_rules_copy = {k: v for k,v in cancelable_rules_copy.items() if string_to_rule(k) != rule}
        new_sorted = sorted(cancelable_rules_copy.keys())

        shield.remove_neg_rule(str(rule))
        shield.cancelable_rules = cancelable_rules_copy
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
        rule_chooser_rules_copy.pop()
        enforceable_rules_copy = shield.enforceable_rules.copy()
        enforceable_rules_copy = {k: v for k,v in enforceable_rules_copy.items() if string_to_rule(k) != rule}
        new_sorted = sorted(enforceable_rules_copy.keys())

        shield.remove_pos_rule(str(rule))
        shield.enforceable_rules = enforceable_rules_copy
        rule_chooser.set_rules_list(rule_chooser_rules_copy)
        rule_chooser.sorted_enforceable_rules = new_sorted

        log.info(f"Removed rule: {rule}")

        rules_snapshot = RuleSnapshot(
            enforceable_rules = enforceable_rules_copy,
            cancelable_rules = shield.cancelable_rules,
        )
        return rules_snapshot


def add_rule(rule, shield, rule_chooser, rule_str = None):
    """ Adds a rule the shield and rulec chooser

    Args:
        rule (Rule): rule to remove
        shield (AspShield): shield to remove the rule from
        rule_chooser (RuleChooser): rule chooser to remove the rule from
        rule_str (string, optional): rule string (will then be converted into a rule object, if rule is not given). Defaults to None.

    Returns:
        RuleSnapshot: snapshot of the new rule set
    """
    if rule_str is not None:
        rule = string_to_rule(rule_str, False)

    rule.mined = False
    
    if rule.polarity == False:
        cancelable_rule = dict()
        cancelable_rule[str(rule)] = [rule]

        cancelable_rules_copy = shield.cancelable_rules.copy()
        cancelable_rules_copy.update(cancelable_rule)

        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.append(len(rule_chooser.rules_list))

        new_sorted = sorted(cancelable_rules_copy.keys())

        shield.add_neg_rule(rule)
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

        enforceable_rules_copy = shield.enforceable_rules.copy()
        enforceable_rules_copy.update(enforcable_rule)

        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.append(len(rule_chooser.rules_list))

        new_sorted = sorted(enforceable_rules_copy.keys())

        shield.add_pos_rule(rule)
        shield.enforceable_rules = enforceable_rules_copy
        rule_chooser.set_rules_list(rule_chooser_rules_copy)
        rule_chooser.sorted_enforceable_rules = new_sorted

        log.info(f"Added rule: {rule}")

        rules_snapshot = RuleSnapshot(
            enforceable_rules = enforceable_rules_copy,
            cancelable_rules = shield.cancelable_rules,
        )
        return rules_snapshot
    

