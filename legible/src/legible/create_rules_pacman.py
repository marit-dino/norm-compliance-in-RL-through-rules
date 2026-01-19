import os
import sys

from rule_learning.util import load_pickle, save_pickle
from shield.create_rules_common import remove_feat, create_new_rules
from shield.relations_pacman import turn_90_4_ghosts, remove_feat_4_ghosts, safe_guard_4_ghosts, turn_90_2_ghosts, \
    remove_feat_2_ghosts, safe_guard_2_ghosts
from shield.rule_classes import string_to_rule, Fact, Rule, RuleHead, RuleBody


def turn(to_turn, turn_dict):
    turned = [to_turn]
    for i in range(3):
        last = turned[-1]
        turned.append(turn_dict[last])
    return turned


def create_rules_pac(rule_str, domain_knowledge):
    rule = string_to_rule(rule_str)
    rule_polarity = rule.polarity
    remove_feat_list = domain_knowledge["remove_feat"]
    action = rule.rule_head.action
    conditions = list(rule.rule_body.conditions)
    for feat in remove_feat_list:
        remove_feat(conditions, feat)
    turned_actions = turn(action,domain_knowledge["turn_90_clockwise"][0])
    turned_cond_list = []
    turn_dict_feat = domain_knowledge["turn_90_clockwise"][1]
    for cond in conditions:
        to_turn_feat = cond.feature
        valuation = cond.valuation
        if to_turn_feat in turn_dict_feat:
            turned_feat = turn(to_turn_feat,turn_dict_feat)
            turned_conds = list(map(lambda feat : Fact(feat,valuation),turned_feat))
        else:
            turned_conds = [cond] * 4 # invariant otherwise
        turned_cond_list.append(turned_conds)
    new_rules = []
    for index,action in enumerate(turned_actions):
        conds = list(map(lambda t_conds : t_conds[index],turned_cond_list))
        if rule_polarity:
            conds.append(domain_knowledge["safe_guard"][action])
        new_rule = Rule(rule_polarity,RuleHead(action),RuleBody(conds))
        new_rules.append(new_rule)
    return new_rules


def generalize_rules(shield,shield_name,nr_ghosts, exact_model_number = None):
    domain_knowledge = {}

    if nr_ghosts == 4:
        domain_knowledge["turn_90_clockwise"] = turn_90_4_ghosts
        domain_knowledge["remove_feat"] = remove_feat_4_ghosts
        domain_knowledge["safe_guard"] = safe_guard_4_ghosts
    elif nr_ghosts == 2:
        domain_knowledge["turn_90_clockwise"] = turn_90_2_ghosts
        domain_knowledge["remove_feat"] = remove_feat_2_ghosts
        domain_knowledge["safe_guard"] = safe_guard_2_ghosts
    else:
        raise Exception("Unknown number of ghosts")

    mutated_shield = create_new_rules(shield, domain_knowledge,
                                      create_rules = lambda rule_str, domain_knowledge : create_rules_pac(rule_str,domain_knowledge))
    mutated_shield_name = shield_name.replace("uncorr", "improved")

    if exact_model_number is None:
        save_pickle(mutated_shield_name,mutated_shield)
    else:
        mutated_shield_name = f"{mutated_shield_name}" #dont need to add exact model number because it is already in there
        save_pickle(mutated_shield_name, mutated_shield, exact_match=True)

    return mutated_shield,mutated_shield_name

if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    mode = sys.argv[3]
    nr_features = int(sys.argv[4])
    algo = sys.argv[5]
    corr = sys.argv[6] == "True"
    corr_string = "corr" if corr else "uncorr"

    exact_model_number = None
    for arg in sys.argv:
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))

    shield_name = f"pickles/shields/{corr_string}/{algo}_{env_name}_{mode}_feat_{nr_features}_{steps}_shield"

    if exact_model_number is None:
        shield = load_pickle(shield_name)
    else:
        shield_name = f"{shield_name}_{exact_model_number}.pkl"
        shield = load_pickle(shield_name,  exact_match=True)

    if "original" in mode:
        nr_ghosts = 4
    elif "small" in mode or "medium" in mode:
        nr_ghosts = 2
    else:
        raise Exception("Unknown env.")

    generalize_rules(shield,shield_name,nr_ghosts,exact_model_number = exact_model_number)