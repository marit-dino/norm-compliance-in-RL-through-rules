import random
import re
import subprocess

import numpy as np
from pandas import DataFrame

import wittgenstein as lw
import pandas as pd
from itertools import combinations

def replace_single_cont_value(v, feature_index, enum_fint_for_index):
    for index, (lower,upper) in enum_fint_for_index:
        if lower <= v <= upper: # treat as closed intervals, because some intervals created by LIME would be empty, like (0,0]
            return index
    print(f"Did not find interval for {v} at index {feature_index}")
    return len(enum_fint_for_index)

def discretize_np_state(s, feature_indices,feature_intervals,categorical_features):
    for i,fi in enumerate(feature_indices):
        if fi not in categorical_features:
            s[i] = replace_single_cont_value(s[i],fi,list(enumerate(feature_intervals[fi])))
    return s

def replace_values_in_series(s, feature_indices, feature_intervals,categorical_features):
    # s is pandas series
    feature_name = s.name
    feature_index = int(feature_name.replace("f",""))
    if feature_index in categorical_features:
        return s
    else:
        enum_fint_for_index = list(enumerate(feature_intervals[feature_index]))
        return s.apply(lambda v : replace_single_cont_value(v, feature_index, enum_fint_for_index))

def discretize_with_intervals(X, feature_indices, feature_intervals,categorical_features):
    return X.apply(lambda s : replace_values_in_series(s,feature_indices,feature_intervals,categorical_features))


def extract_features_from_obs(data, feature_indices,feature_intervals,categorical_features):
    # data is of shape samples x 1 (label) + features
    target = list(data[:,0].astype(int))
    print(data.shape)
    # create df with all the features and then filter by the selected ones
    X = pd.DataFrame(data[:,1:], columns=[f'f{i}' for i in range(0,len(data[0,1:]))]
                      )[[f'f{idx}' for idxs in feature_indices for idx in idxs]]
    
    print("Discretizing")
    X = discretize_with_intervals(X, feature_indices, feature_intervals,categorical_features)
    X = X.apply(lambda x: x.astype(int))

    print("Features created")
    return X, target


def rulesRipper2asp(ruleset, y_label, negated):
    """
    translates rules from RIPPER to a string with ASP format
    """
    
    rules_str = ""
    for rule in ruleset:
        body = [f'{cond.feature}({cond.val})' for cond in rule.conds]
        rules_str += ("-" if negated else "") + f'action({y_label}) :- '
        for atom in body:
            rules_str += atom + (", " if atom != body[-1] else ".\n")
    return rules_str

def select_data_for_neg(X, target,obs_data_min_q_index,action):
    neg_indexes_for_action = list(obs_data_min_q_index[action]) # copy the list
    incorrect_q_min = 0
    for i, action_in_target in enumerate(target):
        if action_in_target == action and i in neg_indexes_for_action:
            neg_indexes_for_action.remove(i)
            incorrect_q_min += 1
    print(f"Removed {incorrect_q_min} data points for incorrect q estimations")
    pos_indexes_for_action = [i for i,act in enumerate(target) if act == action]
    X_result = X.iloc[pos_indexes_for_action + neg_indexes_for_action] #np.concatenate(pos_X,neg_X)
    y = ([False] * len(pos_indexes_for_action)) + [True] * len(neg_indexes_for_action) # must be inverted
    return X_result,y

def getRulesRipper(X, target,feature_importances, actions, prune_size, k,negated,obs_data_min_q_index=None):
    """
    gets all rules for each negated action
    """
    negated = negated
    rules = ""

    for action in actions:
        y = [not negated if y==action else negated for y in target]
        #if there is only one action in this dataset, we skip it
        if sum(y) == 0 or sum(y) == len(y): continue
        if obs_data_min_q_index is not None and negated:
            nr_neg_indexes_for_act = len(obs_data_min_q_index[action])
            print(f"Number of neg data for action{action}: {nr_neg_indexes_for_act}")
            if nr_neg_indexes_for_act > 0:
                X_rules,y_rules = select_data_for_neg(X,target,obs_data_min_q_index,action)
            else:
                X_rules, y_rules = X, y
        else:
            X_rules, y_rules = X,y
        # remove ripper discretization
        max_distinct_features = 0
        for col in X_rules.columns:
            max_distinct_features = max(max_distinct_features,len(X_rules[col].unique()))
        ripper_clf = lw.RIPPER(k=k, dl_allowance=128, prune_size=prune_size, verbosity=0,feature_importances=feature_importances[(action,negated)],
                               n_discretize_bins=max_distinct_features)
        ripper_clf.fit(X_rules, y=y_rules)
        rules += rulesRipper2asp(ripper_clf.ruleset_, action, negated)
    return rules
    
def simplifyContainedRules(rules):
    """
    returns a string in ASP format with the rules after removing the rules 
    that are contained in other rules. For instance, if we have the following 
    two rules,
    action(continue) :- tile_3_5(e), tile_2_6(s).
    action(continue) :- tile_3_5(e), tile_2_6(s), tile_4_6(l).
    we just need the first one since the second one is contained in the first one
    """
    
    rules_splitted = []
    for rule in rules:
        if ":-" in rule:
            head, body = rule.replace("\n","").split(":-")
            head = head.replace(" ", "")
            new_body = []
            for predicate in body.split(","):
                if "not " in predicate:
                    predicate = "not " + predicate.replace("not ","").replace(" ","")
                else:
                    predicate = predicate.replace(" ", "")
                new_body.append(predicate)
            rules_splitted.append([head] + new_body)
    
    n_rules_removed = 0
    for i in range(len(rules_splitted)): # all rules
        found = False # if a simplified rule is found, then go to the next rule
        rule = rules_splitted[i-n_rules_removed]
        for j in range(1,len(rule[1:])): # all combinations of all lengths
            for comb in combinations(rule[1:],j):
                # look for a smaller set in all rules
                for rule_comp in rules_splitted: #TODO: remove current rule
                    # the body of the simplified rule is smaller than the original rule
                    # and the head of the rule is the same
                    if (len(rule_comp[1:]) <= j) and (rule_comp[0]==rule[0]) and (
                            set(comb) == set(rule_comp[1:])):
                        rules_splitted.pop(i-n_rules_removed)
                        n_rules_removed += 1
                        found = True
                        break
                if found: break
            if found: break
    
    simplified_rulesASP = ""
    for rule in rules_splitted:
        simplified_rulesASP += rule[0] + " :- " + ", ".join(rule[1:]) + "."
    
    return simplified_rulesASP.split(".")[:-1]

def evaluate_rule_list(asp_rules, X, target):
    '''
    for each rule, it returns 
    .the number of examples that satisfy the conditions of that rule (coverage),
    .the number of examples that satisfy the conditions of that rule and the
    head of the rule is the same as in the example (right)
    .the number of examples that satisfy the conditions of that rule and the
    head of the rule is not the same as in the example (wrong)
    '''
    df = X.copy()
    df['action'] = target
    rights = []
    coverages = []
    for rule in asp_rules:
        right = 0
        coverage = 0
        if rule.split(':-')[1].replace(" ", "") == "":
            print("Wrong rule: " + rule)
            coverages.append(-1)
            rights.append(0)
            continue
        for example in df.to_numpy():

            found = True # if a condition is not satisfied, it will be false and the loop will be exited
            # check if the conditions are satisfied
            for predicate in rule.split(':-')[1].split(','):
                try:
                    if predicate.strip()[:len("not ")] == "not ":
                        feature = predicate.strip()[len("not "):predicate.strip().find('(')]
                        if int(predicate[predicate.find('(')+1:predicate.find(")")]) == int(example[df.columns.to_list().index(feature)]):
                            found = False
                            break
                    else:
                        feature = predicate.strip()[:predicate.strip().find('(')]
                        if int(predicate[predicate.find('(')+1:predicate.find(")")]) != int(example[df.columns.to_list().index(feature)]):
                            found = False
                            break
                except Exception as e:
                    print("Issues found during rule evaluation:")
                    print(f"Predicate: {predicate}")
                    print(f"Rule: {rule}")
                    print(f"Exception: {e}")

            if found:
                coverage += 1
                # if the head of the rule is the same as the original class
                head = rule.split(':-')[0]
                eq = int(example[df.columns.to_list().index('action')]) == int(head[head.find('(')+1:head.find(')')])

                if head.strip()[0] == "-":
                    if not eq:
                        right += 1
                else:
                    if eq:
                        right += 1
        coverages.append(coverage)
        rights.append(right)
    accuracies = [rights[i]/coverages[i] if coverages[i] != 0 else 0 for i in range(len(rights))]
    return coverages, rights, accuracies

def get_metrics(rules, X, target):
    ''' 
    It takes the rules as a single string in ASP format, and a dataframe with
    the examples where the rules are going to be tested.
    It returns a list with their accuracy and coverage
    '''
    # quality of the rules
    coverages, rights, accuracies = evaluate_rule_list(rules, X, target)
    rules_metrics = [[rules[i], accuracies[i], 
                      coverages[i]/len(target)] for i in range(len(coverages))]
    return [rule[1:] for rule in rules_metrics]

def create_arf_string(X : DataFrame, target):
    arf_string = "@relation test \n"
    for col in X.columns:
        unique_vals_col = X[col].drop_duplicates().values
        unique_vals_col = ",".join(map(lambda val : f"'{val}'",unique_vals_col))
        arf_string += f"@attribute {col} {{ {unique_vals_col} }}\n"

    actions = ",".join(map(lambda val: f"'{val}'", set(target)))
    arf_string += f"@attribute 'Class' {{ {actions} }}\n"
    arf_string += "@data \n"
    for row_nr in range(X.shape[0]):
        row_str = ",".join(map(lambda val : f"'{val}'",X.iloc[row_nr].values))
        action = target[row_nr]
        arf_string += f"{row_str}, '{action}'\n"

    return arf_string

def get_rules(X, target, feature_importances,actions, obs_data_min_q_index=None,
              negated=True, algorithm='ripper', K=5, PRUNE_SIZE=0.33,
              MAX_RULE_LENGTH = 3, BEAM_WIDTH = 5, MIN_ACC = 0.9, MIN_COV = 0.1):
    '''
    Parameters for RIPPER
    ----------
    K : The default is 0.
    PRUNE_SIZE : The default is 0.

    Parameters for CN2
    MAX_RULE_LENGTH : The default is 3.
    BEAM_WIDTH : The default is 5.
    
    ----------
    Parameters for both
    ----------
    MIN_ACC. 1.0 to keep only the rules without a counterexample in the dataset.
        The deafult is 0.9.
    MIN_COV. Minimum coverage. Percentage. The default is 0.1.
    
    algorithm : either ripper or cn2. The default is ripper.
    X : dataframe with the value of the features in the examples.
    target : list of ints with the value of the action taken by the agent in each example
    actions : list with the possible actions.
    
    Returns
    -------
    string with the rules in ASP format

    '''
    if algorithm == "ripper":
        rulesASP = getRulesRipper(X, target,feature_importances, actions,
                                  PRUNE_SIZE, K,negated=negated,obs_data_min_q_index=obs_data_min_q_index)
    else: # currently we only support RIPPER
        raise NotImplementedError

    # convert string of rules to list
    rulesASPlist = list(filter(lambda x: (x!=""), rulesASP.replace("\n","").split('.')))
    # get accuracy and coverage for every rule
    metrics = get_metrics(rulesASPlist, X, target)
    
    # filter by minimum accuracy and coverage
    rules_and_metrics = [[rulesASPlist[i], metrics[i]] for i in range(len(rulesASPlist)) if (
        (metrics[i][0]>=MIN_ACC) and (metrics[i][1]>=MIN_COV))]
    
    # unpack previous list into two lists
    filteredRules = [rules_and_metrics[i][0] for i in range(len(rules_and_metrics))]
    metrics = [rules_and_metrics[i][1] for i in range(len(rules_and_metrics))]
    
    # remove unnecessary rules
    cleanRules = simplifyContainedRules(filteredRules)
    print("Filtered")
    print(filteredRules)
    
    filteredRulesAndMetrics = [[rule, metrics[filteredRules.index(rule)]] for rule in cleanRules]
    filteredRulesAndMetrics.sort(key = lambda x : x[1][1], reverse=True)
    return filteredRulesAndMetrics

    

    