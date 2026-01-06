import logging, copy
from gym_pacman_rules.envs.featureExtractors import features_dict_to_array
from legible.rule_learning.util import save_pickle
from legible.shield.shields import RuleChooser 
from legible.create_rules_pacman import string_to_rule



log = logging.getLogger(__name__)



class RuleSnapshot:
    enforceable_rules = dict()
    cancelable_rules = dict()
    rule_indices = []

    def __init__(self, enforceable_rules, cancelable_rules):
        self.enforceable_rules = enforceable_rules 
        self.cancelable_rules = cancelable_rules  
        self.rule_indices = range(0, len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))



def collect_data(model, env, action_tensor,feature_extractor, shield, rule_chooser,training_algorithm):
    data = []

    obs, _info = env.reset()
    policy = model.policy
    obs_t, _vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)

    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )

    #TODO fix number in cfg
    for i in range(15):
        while True:
            action, _states = model.predict(obs)
            q_values = policy.q_net(obs_t).squeeze()
            act_logits = q_values        
            obs_rules = features_dict_to_array(feature_extractor.getFeatures(env.unwrapped.game.state,action))
            
            data.append(obs_rules)

            action, _triggered_rules = get_action(model, obs, obs_rules, shield, env.unwrapped.game.state,rule_chooser, rules_snapshot, training_algorithm, act_logits, action_tensor)
            
            obs, _reward, term, trunc, _info = env.step(action)

            obs_t, _vectorized_env = policy.obs_to_tensor(obs)
            obs_t = obs_t.to(action_tensor.device)

            if term or trunc:
                obs, _info = env.reset()
                obs_t, _vectorized_env = policy.obs_to_tensor(obs)
                obs_t = obs_t.to(action_tensor.device)
                break

    return data
        

def prune_rule_set(model, env, action_tensor, feature_extractor, shield, rule_chooser, cfg):
    data = collect_data(model, env, action_tensor, feature_extractor, shield, rule_chooser, cfg.training.algorithm)
    rules_snapshot = remove_unused_rules(data, rule_chooser, shield)
    #prune_rule_conditions()

    return rules_snapshot

def optimize_rule_set():
    print("TODO")

def remove_unused_rules(data, rule_chooser, shield):
    mined_rules = [r for r in (rule_chooser.sorted_cancel_rules + rule_chooser.sorted_enforce_rules) if r.mined]
    flattened_data = [r for rs in data for r in rs]

    for r in mined_rules:
        if r not in flattened_data:
            rules_snapshot = remove_rule(r, shield, rule_chooser)
            log.info(f"remove rule : {r} based on {flattened_data}")
            log.info(f"removed unused rule {r}")

    return rules_snapshot

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


def get_action(model, obs, obs_rules, shield, state, rule_chooser, rules_snapshot, algo_name, act_logits, action_tensor):
    action, _state = model.predict(obs)
    triggers,triggered_actions, triggered_rules = shield.does_rule_trigger(obs_rules,rule_chooser,rules_snapshot)
    
    from legible.evaluate_policy import change_action

    if triggers:
        (pos_actions_triggered, neg_actions_triggered) = triggered_actions
        changed_action, activated_created_rules = change_action(action,pos_actions_triggered,neg_actions_triggered,algo_name,act_logits, action_tensor,
                                        triggered_rules, change_type="favor_enforce")
        if changed_action is not None:
            if len(activated_created_rules) > 0:
                log.info(f"Updated rule(s) used:\n {[str(r[0]) for r in activated_created_rules]}\nin state\n{state}")
            action = changed_action
            
    return action, triggered_rules


def remove_rule(rule, shield, rule_chooser):
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
        rule_chooser_rules_copy.remove(-1)
        enforcable_rules_copy = shield.enforceable_rules.copy()
        enforcable_rules_copy = {k: v for k,v in enforcable_rules_copy.items() if string_to_rule(k) != rule}
        new_sorted = sorted(enforcable_rules_copy.keys())

        shield.remove_pos_rule(str(rule))
        shield.enforceable_rules = enforcable_rules_copy
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

        enforcable_rules_copy = shield.enforceable_rules.copy()
        enforcable_rules_copy.update(enforcable_rule)

        rule_chooser_rules_copy = rule_chooser.rules_list.copy()
        rule_chooser_rules_copy.append(len(rule_chooser.rules_list))

        new_sorted = sorted(enforcable_rules_copy.keys())

        shield.add_pos_rule(rule)
        shield.enforceable_rules = enforcable_rules_copy
        rule_chooser.set_rules_list(rule_chooser_rules_copy)
        rule_chooser.sorted_enforceable_rules = new_sorted

        log.info(f"Added rule: {rule}")

        rules_snapshot = RuleSnapshot(
            enforceable_rules = shield.enforceable_rules,
            cancelable_rules = cancelable_rules_copy,
        )
        return rules_snapshot

