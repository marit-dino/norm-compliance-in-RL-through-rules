import copy
import re
from typing import List


class Fact:
    def __init__(self, feature, valuation):
        self.feature = feature
        self.valuation = valuation

    def __eq__(self, other):
        if isinstance(other, Fact):
            return self.feature == other.feature and self.valuation == other.valuation
        return False

    def __hash__(self):
        return hash((self.feature,self.valuation))

    def __str__(self):
        return f"f{self.feature}({self.valuation})"


class RuleBody:
    def __init__(self, conditions: List[Fact]):
        # tuple to make the conditions hashable and immutable
        self.conditions = tuple(sorted(conditions,key=lambda cond : cond.feature))

    def __eq__(self, other):
        if isinstance(other, RuleBody):
            return self.conditions == other.conditions
        return False

    def __hash__(self):
        return hash(self.conditions)

    def __str__(self):
        return ", ".join(map(str,self.conditions))


class RuleHead:
    def __init__(self, action):
        self.action = action

    def __eq__(self, other):
        if isinstance(other, RuleHead):
            return self.action == other.action
        return False

    def __hash__(self):
        return hash(self.action)

    def __str__(self):
        return f"action({self.action})"


class Rule:
    def __init__(self, polarity : bool,rule_head : RuleHead, rule_body : RuleBody, mined = True):
        self.polarity = polarity
        self.rule_head = rule_head
        self.rule_body = rule_body
        self.mined = mined

    def __eq__(self, other):
        if isinstance(other, Rule):
            return (self.polarity == other.polarity and self.rule_head == other.rule_head and
                    self.rule_body == other.rule_body)
        return False

    def __hash__(self):
        return hash((self.polarity,self.rule_body,self.rule_head))

    def add_feature(self,feature,feature_valuation):
        mutated_body_conds = list(copy.deepcopy(self.rule_body.conditions))
        mutated_body_conds.append(Fact(feature,feature_valuation))
        return Rule(self.polarity,self.rule_head, RuleBody(mutated_body_conds))

    def __str__(self):
        rule_str_start = "-" if not self.polarity else ""
        return f"{rule_str_start}{self.rule_head} :- {self.rule_body}."


def parse_body_string(body_str) -> RuleBody:
    body_str_split = body_str.split(",")
    body = []
    for cond_str in body_str_split:
        cond_str = cond_str.strip()
        cond_matcher = re.match(r'f(\d+)\((-?\d+)\)',cond_str)
        if cond_matcher is None:
            print(cond_str)
        matched_groups = cond_matcher.groups()
        feature = int(matched_groups[0])
        valuation = int(matched_groups[1])
        body.append(Fact(feature,valuation))
    return RuleBody(body)


def string_to_rule(rule_string : str, mined = False) -> Rule :
    polarity = False if rule_string.startswith("-") else True
    if not polarity:
        rule_string = rule_string[1:]
    rule_str_split = rule_string.split(":-")
    head_str = rule_str_split[0].strip()
    body_str = rule_str_split[1].strip()
    head_match = re.search(r"\d+", head_str)
    action = int(head_match.group())
    body = parse_body_string(body_str)
    return Rule(polarity,RuleHead(action),body, mined)


def contains_fact(rule : Rule, feature, feature_valuation):
    fact = Fact(feature,feature_valuation)
    return fact in rule.rule_body.conditions


def copy_rule_without(rule, feature, feature_valuation):
    fact_to_exclude = Fact(feature,feature_valuation)
    mutated_rule_body = RuleBody([fact for fact in rule.rule_body.conditions if fact != fact_to_exclude])
    return Rule(rule.polarity,rule.rule_head,mutated_rule_body)


def copy_rule_with_replacement(rule, feature_in_rule, feature_replacement, feature_valuation):
    rule = copy_rule_without(rule,feature_in_rule,feature_valuation)
    rule = rule.add_feature(feature_replacement,feature_valuation)
    return rule
