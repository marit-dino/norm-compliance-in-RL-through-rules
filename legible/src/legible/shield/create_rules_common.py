from typing import List

from shield.rule_classes import Fact, string_to_rule, Rule
from shield.shields import AspShield


def remove_feat(condition_list : List[Fact], feature):
    contained_fact = [index for index,fact in enumerate(condition_list) if fact.feature == feature]
    if len(contained_fact) > 0:
        condition_list.pop(contained_fact[0])


def remove_one_duplicate_rule(rules_list):
    for i in range(len(rules_list)):
        for j in range(i+1,len(rules_list)):
            rule_pair1 = rules_list[i]
            rule_pair2 = rules_list[j]
            base_rule1 = string_to_rule(rule_pair1[0])
            rule_set2 = rule_pair2[1]
            for rule in rule_set2:
                if base_rule1 == rule:
                    print(f"Due to {base_rule1} == {str(rule)}")
                    print(f"Removed {[str(r) for r in rule_set2]}")
                    rules_list.pop(j) # I don't know if in place is good but it should be fine
                    return rules_list
    return rules_list


def remove_duplicate_rules(rules_dict :  dict[str,List[Rule]]):
    rules_list = list(rules_dict.items())
    nr_rules = len(rules_list)
    print(f"Initial nr. rules: {nr_rules}")
    while True:
        rules_list = remove_one_duplicate_rule(rules_list)
        new_nr_rules = len(rules_list)
        if new_nr_rules == nr_rules:
            break
        nr_rules = new_nr_rules
    print(f"Final nr. rules: {new_nr_rules}")
    return dict(rules_list)


def turn_rules_to_str(rules_dict :  dict[str,List[Rule]]):
    return {k : [str(r) for r in rule_list] for k,rule_list in rules_dict.items()}


def create_new_rules(shield : AspShield, domain_knowledge, create_rules,do_remove=True):
    enforceable_rules = dict()
    cancelable_rules = dict()
    for pos_rule in shield.pos_rules_list:
        if len(pos_rule.strip()) == 0:
            continue
        new_enforceable_rules = create_rules(pos_rule, domain_knowledge)
        if type(new_enforceable_rules) == dict:
            for rule_type in new_enforceable_rules.keys():
                new_enforceable_rules_list = new_enforceable_rules[rule_type]
                pos_rule_str = f"{pos_rule} [{rule_type}]"
                enforceable_rules[pos_rule_str] = new_enforceable_rules_list
                print_new_rules(new_enforceable_rules_list, pos_rule_str)
        else:
            enforceable_rules[pos_rule] = new_enforceable_rules
            print_new_rules(new_enforceable_rules, pos_rule)
    for neg_rule in shield.neg_rules_list:
        if len(neg_rule.strip()) == 0:
            continue
        new_cancelable_rules = create_rules(neg_rule, domain_knowledge)
        if type(new_cancelable_rules) == dict:
            for rule_type in new_cancelable_rules.keys():
                new_cancelable_rules_list = new_cancelable_rules[rule_type]
                neg_rule_str = f"{neg_rule} [{rule_type}]"
                cancelable_rules[neg_rule_str] = new_cancelable_rules_list
                print_new_rules(new_cancelable_rules_list, neg_rule_str)
        else:
            cancelable_rules[neg_rule] = new_cancelable_rules
            print_new_rules(new_cancelable_rules, neg_rule)
    if do_remove:
        enforceable_rules = remove_duplicate_rules(enforceable_rules)
        cancelable_rules = remove_duplicate_rules(cancelable_rules)
    shield.enforceable_rules = turn_rules_to_str(enforceable_rules) if len(enforceable_rules) > 0 else dict()
    shield.cancelable_rules = turn_rules_to_str(cancelable_rules) if len(cancelable_rules) > 0 else dict()
    return shield


def print_new_rules(new_enforceable_rules, pos_rule):
    print("**" * 40)
    print(f"Base rule: {pos_rule}")
    print("Created")
    for rule in new_enforceable_rules:
        print(rule)
