import json
import random
import sys
from collections import defaultdict

from rule_learning.util import load_pickle, save_pickle
from shield.create_rules_common import turn_rules_to_str
from shield.rule_classes import string_to_rule, Fact, RuleBody, Rule, RuleHead
from shield.shields import AspShield


def find_used_facts(all_rules):
    available_facts = defaultdict(list)
    for rule in all_rules:
        for fact in rule.rule_body.conditions:
            available_facts[fact.feature].append(fact.valuation)
    return available_facts


def find_rule_len_cnts(all_rules):
    rule_len_cnts = defaultdict(int)
    for rule in all_rules:
        rule_len_cnts[len(rule.rule_body.conditions)] += 1

    return sorted(list(rule_len_cnts.items()), key=lambda len_cnt : len_cnt[0])


def find_rule_pol_cnts(all_rules):
    rule_pol_cnts = defaultdict(bool)
    for rule in all_rules:
        rule_pol_cnts[rule.polarity] += 1

    return sorted(list(rule_pol_cnts.items()), key=lambda len_cnt : len_cnt[0])


def create_random_rule(actions, polarity, length, available_features,available_facts):
    action = random.choice(actions)
    random_conds = []
    features_copy = list(available_features)
    for i in range(length):
        feat = random.choice(features_copy)
        features_copy.remove(feat)
        feat_val = random.choice(available_facts[feat])
        random_conds.append(Fact(feat,feat_val))
    return Rule(polarity,RuleHead(action),RuleBody(random_conds))

def create_random_group(actions,polarity, length, group_size,available_features,available_facts):
    group = []
    for i in range(group_size):
        group.append(create_random_rule(actions,polarity,length,available_features,available_facts))
    return group

def create_random_rules(nr_rules, actions, available_facts, rule_len_cnts, polarity_cnts,rule_group_size):
    enforceable_rules = dict()
    cancelable_rules = dict()
    available_features = list(available_facts.keys())
    group_size_min = rule_group_size[0]
    group_size_max = rule_group_size[1]
    lens,len_weights = tuple(zip(*rule_len_cnts))
    pols,pol_weights = tuple(zip(*polarity_cnts))

    for i in range(nr_rules):
        polarity = random.choices(pols,weights=pol_weights,k=1)[0]
        length = random.choices(lens,weights=len_weights,k=1)[0]
        group_size = random.randint(group_size_min,group_size_max)
        random_group = create_random_group(actions,polarity,length,group_size,available_features,available_facts)
        if polarity:
            rule_name = f"Positive random rule {len(enforceable_rules) + 1}"
            enforceable_rules[rule_name] = random_group
        else:
            rule_name = f"Negative random rule {len(cancelable_rules) + 1}"
            cancelable_rules[rule_name] = random_group
        print("**"*20)
        print({rule_name})
        for g in random_group:
            print(str(g))
    return enforceable_rules,cancelable_rules


def create_store_random_shield(shield : AspShield, shield_name,rule_group_size,exact_model_number=None):
    pos_rules = [string_to_rule(rule_str) for rule_str in shield.pos_rules_list if len(rule_str.strip()) > 0]
    neg_rules = [string_to_rule(rule_str) for rule_str in shield.neg_rules_list if len(rule_str.strip()) > 0]
    all_rules = pos_rules + neg_rules
    available_facts = find_used_facts(all_rules)
    rule_len_cnts = find_rule_len_cnts(all_rules)
    polarity_cnts = find_rule_pol_cnts(all_rules)
    
    mutated_shield_name = shield_name.replace("uncorr", "random")

    enforceable_rules, cancelable_rules = create_random_rules(len(all_rules), shield.actions,available_facts,
                                                              rule_len_cnts,polarity_cnts,rule_group_size)
    mutated_shield = shield

    mutated_shield.enforceable_rules = turn_rules_to_str(enforceable_rules) if len(enforceable_rules) > 0 else dict()
    mutated_shield.cancelable_rules = turn_rules_to_str(cancelable_rules) if len(cancelable_rules) > 0 else dict()
    if exact_model_number is None:
        save_pickle(mutated_shield_name, mutated_shield)
    else:
        mutated_shield_name = f"{mutated_shield_name}"
        save_pickle(mutated_shield_name, mutated_shield, exact_match=True)
    return mutated_shield, mutated_shield_name


if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    mode = sys.argv[3]
    nr_features = int(sys.argv[4])
    algo = sys.argv[5]
    corr = sys.argv[6] == "True"
    rule_group_size = json.loads(sys.argv[7]) # rule_group_size is [min_nr_rules,max_nr_rules]
    corr_string = "corr" if corr else "uncorr"

    if "Berkeley" in env_name:
        shield_name = f"pickles/shields/{corr_string}/{algo}_{env_name}_{mode}_feat_{nr_features}_{steps}_shield"
    else:
        shield_name = f"pickles/shields/{corr_string}/{algo}_{env_name}_0_feat_{nr_features}_{steps}_shield"


    exact_model_number = None
    for arg in sys.argv:
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))
    if exact_model_number is None:
        shield = load_pickle(shield_name)
    else:
        shield_name = f"{shield_name}_{exact_model_number}.pkl"
        shield = load_pickle(shield_name, exact_match=True)

    create_store_random_shield(shield,shield_name,rule_group_size,exact_model_number=exact_model_number)