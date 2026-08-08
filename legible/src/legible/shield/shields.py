import random
from abc import ABC, abstractmethod

import clingo

from rule_learning.learnRules import replace_single_cont_value
from shield.rule_classes import string_to_rule

def getModels(generation, rules, constraint, obs):
    """
    calls Clingo to get the models given the different parts of the logic program:
        generation of cases,
        rules,
        constraint,
        observations.
    all the variables need to be in a string with ASP format
    """

    # We should also add the following constraint but there is noise...
    # it can be too restrictive
    # only1action = ":- action(X), action(Y), X!=Y."
    ctl = clingo.Control()
    ctl.add("all", [], generation + rules + constraint + obs)
    ctl.ground([("all", [])])
    # we do not need all the models, so the following line is not needed
    # ctl.configuration.solve.models="0"

    res = []
    with ctl.solve(yield_=True) as handle:
        for m in handle:
            s = set()
            for a in m.symbols(atoms=True):
                s.add(a)
            res.append(s)
    return res



class RuleChooser:
    def __init__(self, shield):
        self.sorted_enforce_rules = sorted(shield.enforceable_rules.keys())
        self.sorted_cancel_rules = sorted(shield.cancelable_rules.keys())
        self.shield = shield
        self.rules_list = None

    def select_rules(self, rules_snapshot, return_keys=False):
        sorted_enforce_rules = sorted(rules_snapshot.enforceable_rules.keys())
        sorted_cancel_rules = sorted(rules_snapshot.cancelable_rules.keys())
        selected = []
        rule_keys = []       
        for rule_index in rules_snapshot.rule_indices:
            if rule_index < len(sorted_enforce_rules):
                rule_key = sorted_enforce_rules[rule_index]
                rule_keys.append(rule_key)
                selected.extend(rules_snapshot.enforceable_rules[rule_key])
            else:
                rule_key = sorted_cancel_rules[rule_index - len(sorted_enforce_rules)]
                rule_keys.append(rule_key)
                selected.extend(rules_snapshot.cancelable_rules[rule_key])
        if return_keys:
            return selected, rule_keys
        else:
            return selected
        
    def set_rules_list(self,rules_list):
        self.rules_list = rules_list
    def choose_rules(self,rules_snapshot):
        if self.rules_list is None:
            raise Exception("Rules list is not set")
        return self.select_rules(rules_snapshot)

# copied from asp shield
def discretize(fvs, feature_indices, categorical_features, feature_intervals):
    for i, fi in enumerate(feature_indices):
        if fi in categorical_features:
            continue
        else:
            enum_interval = enumerate(feature_intervals[fi])
            fvs[i] = replace_single_cont_value(fvs[i], fi, enum_interval)
    return fvs


def abstract_obs(obs,action, feature_indices,categorical_features, feature_intervals,reward):
    features_values = obs[feature_indices]
    features_values = discretize(features_values,feature_indices,categorical_features, feature_intervals).astype(int).astype(str)
    reward_string = ""
    if reward < 0:
        reward_string = "neg_"
    elif reward > 0:
        reward_string = "pos_"
    return reward_string + "_" + str(action) + "_".join(features_values)


class Shield(ABC):
    def reset(self):
        pass

    def step(self, action, next_state, reward):
        pass

    @abstractmethod
    def get_blocked_actions(self, state):
        pass


class NoopShield(Shield):

    def get_blocked_actions(self, state):
        return []

class AspShield(Shield):
    def __init__(self,num_actions,feature_indices,feature_names,neg_rules,pos_rules,
                 feature_intervals,
                 categorical_features,enforceable_rules = None, cancelable_rules = None):
        self.actions = list(range(num_actions))
        self.feature_indices = feature_indices
        self.feature_names = feature_names
        self.all_neg_rules = neg_rules
        self.all_pos_rules = pos_rules

        # self.neg_rules = self.split_rules_per_action(neg_rules)
        self.pos_rules_list = self.all_pos_rules.split("\n")
        self.neg_rules_list = self.all_neg_rules.split("\n")

        self.feature_intervals = feature_intervals
        self.categorical_features = categorical_features
        self.nr_rules = len(self.pos_rules_list) + len(self.neg_rules_list)
        self.enforceable_rules = enforceable_rules
        self.cancelable_rules = cancelable_rules


    def add_neg_rule(self, rule):
        self.all_neg_rules += f"\n{str(rule)}"
        self.neg_rules_list.append(str(rule))
        self.nr_rules += 1  

    def remove_neg_rule(self, rule_str):
        neg_rules_list_copy = self.neg_rules_list.copy()
        neg_rules_list_copy = [r for r in neg_rules_list_copy if rule_str != str(string_to_rule(r))]
        self.all_neg_rules = "\n".join(neg_rules_list_copy)
        self.neg_rules_list = neg_rules_list_copy
        self.nr_rules -= 1

    def add_pos_rule(self, rule):
        self.all_pos_rules += f"\n{str(rule)}"
        self.pos_rules_list.append(str(rule))
        self.nr_rules += 1

    def remove_pos_rule(self, rule_str):
        pos_rules_list_copy = self.pos_rules_list.copy()
        pos_rules_list_copy = [r for r in pos_rules_list_copy if rule_str != str(string_to_rule(r))]
        self.all_pos_rules = "\n".join(pos_rules_list_copy)
        self.pos_rules_list = pos_rules_list_copy
        self.nr_rules -= 1

    def get_trigger_action(self, rules, facts, original_rules):
        if len(rules) == 0:
            return [], []
        res = getModels(generation="",
                        rules="\n".join(rules),
                        constraint="",
                        obs=" ".join(facts))

        if len(res) == 0:
            return [], []
        else:
            triggered_actions_rules = [item.arguments for item in res[0] if (item.name == "triggered_by")]
            triggered_actions = list(set([int(str(item[1])) for item in triggered_actions_rules]))
            triggered_actions_rules = [self.transform_triggered_rule(rule_action, rules, original_rules) for rule_action in triggered_actions_rules]
            return triggered_actions, triggered_actions_rules

    def transform_rule(self, rule, i):
        action = rule.rpartition("action(")[2].split(")")[0]
        ret1 = f"{rule.strip('.')}, triggered_by({i}, {action})."
        ret2 = f"triggered_by({i}, {action}){rule.strip(f'-action({action})')}"
        return ret1, ret2
    
    def transform_triggered_rule(self, rule_action, rules, original_rules):
        triggered_str = f", triggered_by({rule_action[0]}, {rule_action[1]})."
        rule_str = list(filter(lambda r: r.endswith(triggered_str), rules))[0].rsplit(triggered_str)[0] + "."
        from shield.rule_classes import string_to_rule
        # conversion to rule changes the order of the featuers, so "original_rules[rule_str]" does not work
        return list(filter(lambda r: r[0] == (string_to_rule(rule_str)), original_rules.values()))[0]


    def does_rule_trigger(self,obs, rule_chooser : RuleChooser, rules_snapshot):
        features = self.raw_features(obs)   
        facts = self.raw_features_into_facts(features)

        rules = rule_chooser.choose_rules(rules_snapshot)
        pos_rules = [rt for i, r in enumerate(rules) if str(r).startswith("action") for rt in self.transform_rule(str(r), i)]
        neg_rules = [rt for i, r in enumerate(rules) if str(r).startswith("-action") for rt in self.transform_rule(str(r), i)]

        pos_triggered, pos_rules_triggered = self.get_trigger_action(pos_rules, facts, rules_snapshot.enforceable_rules)
        neg_triggered, neg_rules_triggered = self.get_trigger_action(neg_rules, facts, rules_snapshot.cancelable_rules)
        if len(pos_triggered) == 0 and len(neg_triggered) == 0:
            return False, None, None

        # check for conflicts
        # if two positive actions trigger at the same time, just return negative, or do nothing?
        if len(pos_triggered) > 1:
            return False, (None,neg_triggered), (pos_rules_triggered, neg_rules_triggered)
        # check for conflicts between negative and positive triggers
        for pos_action in pos_triggered:
            if pos_action in neg_triggered:
                return False, None, (pos_rules_triggered, neg_rules_triggered)
        # there can only be one positive
        return True,(pos_triggered[0] if pos_triggered else None,neg_triggered), (pos_rules_triggered, neg_rules_triggered)

    def get_blocked_actions(self, state, rules_snapshot):
        cr = rules_snapshot.cancelable_rules
        features = self.raw_features(state)
        facts = self.raw_features_into_facts(features)
        res = getModels(generation="",
                        rules="\n".join(cr.keys()),
                        constraint="",
                        obs=" ".join(facts))
        # there are no choice or generation rules, so only one model will exist
        # (maybe the empty one)
        if len(res) == 0:
            return []

        neg_actions = [int(str(item.arguments[0])) for item in res[0] if (
                item.name == "action") and (not item.positive)]
        neg_actions.sort()
        # if neg_actions == self.actions:
        #     # print("Tried to block all actions, blocking none")
        #     return []
        # # print(neg_actions)
        return neg_actions

    def discretize(self,fvs):
        fvs = fvs.copy()
        for i,fi in enumerate(self.feature_indices):
            if fi not in self.categorical_features:
                enum_interval = enumerate(self.feature_intervals[fi])
                fvs[i] = replace_single_cont_value(fvs[i], fi, enum_interval) 
        return fvs

    def raw_features(self, obs):
        features_values = obs.copy()
        features_values = self.discretize(features_values).astype(int)

        return features_values

    def raw_features_into_facts(self, features_values):
        facts_list = list(
            map(lambda fi_value: f"f{fi_value[0]}({fi_value[1]}).",
                zip(self.feature_indices, features_values.tolist())))
        return facts_list

    def get_nr_rules(self):
        return self.nr_rules


class RandomShield(Shield):
    def __init__(self,num_actions : int,percent_random : int, cancel : bool, cancel_stop : float = 0.66):
        self.triggers = dict()
        self.num_actions = num_actions
        self.cancel = cancel
        self.cancel_stop = cancel_stop
        percent_list = list(range(100))
        rand_selection = []
        for i in range(percent_random):
            sel_index = random.choice(percent_list)
            rand_selection.append(sel_index)
            percent_list.remove(sel_index)
        for i in rand_selection:
            self.triggers[i] = self.create_trigger()
        for i,t in self.triggers.items():
            print(f"Generated {i}: {t}")

    def create_trigger(self):
        if self.cancel:
            return self.create_cancel_trigger()
        else:
            return self.create_enforce_trigger()

    def does_rule_trigger(self,obs, rule_chooser : RuleChooser):
        hash_val = hash(str(obs)) % 100
        if hash_val in self.triggers:
            if self.cancel:
                return True, (None,self.triggers[hash_val])
            else:
                return True, (self.triggers[hash_val],[])
        else:
            return False, None

    def get_blocked_actions(self, state):
        raise Exception("Not implemented")

    def get_nr_rules(self):
        pass

    def create_enforce_trigger(self):
        return random.randint(0,self.num_actions-1)

    def create_cancel_trigger(self):
        canceled = []
        choices = list(range(self.num_actions))
        for i in range(self.num_actions-1):
            action = random.choice(choices)
            choices.remove(action)
            canceled.append(action)
            if random.random() < self.cancel_stop:
                break
        return canceled
