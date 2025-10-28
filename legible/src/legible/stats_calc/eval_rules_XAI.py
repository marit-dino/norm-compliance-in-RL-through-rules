import glob
from collections import defaultdict

from rule_learning.util import load_pickle
from shield.rule_classes import string_to_rule, Rule, RuleBody

def to_nice_name(exp_name):
    level = exp_name[exp_name.index("v0") + len("v0_") :exp_name.index("_feat")]
    level = level.replace("Classic","")
    level = level.replace("_no_capsules", "(NC)")
    return level


def contains_features_from(rule_body : RuleBody, features):
    rule_features = set(map(lambda cond: cond.feature,rule_body.conditions))
    return len(rule_features.intersection(features)) > 0


def cnt_mention(rules_for_exp : list[Rule], type_of_feature, exp_name):
    feature_list = []
    if type_of_feature == "capsules":
        feature_list.extend(list(range(2,8)))
    elif type_of_feature == "food":
        feature_list.extend(list(range(8, 14)))
    elif type_of_feature == "ghosts":
        feature_list.extend(list(range(0, 2)))
        if "original" not in exp_name:
            feature_list.extend(list(range(14, 62)))
        else:
            feature_list.extend(list(range(14, 110)))
    elif type_of_feature == "ghosts-scared":
        feature_list.extend(list(range(0, 2)))
        if "original" not in exp_name:
            feature_list.extend([34,35,58,59])
        else:
            feature_list.extend([34,35,58,59,82,83,106,107])

    elif type_of_feature == "position":
        if "original" not in exp_name:
            feature_list.extend(list(range(62, 69)))
        else:
            feature_list.extend(list(range(110, 117)))
    cnt_pos = 0
    cnt_neg = 0
    features = set(feature_list)
    for r in rules_for_exp:
        if contains_features_from(r.rule_body,features):
            if r.polarity:
                cnt_pos += 1
            else:
                cnt_neg += 1
    return cnt_pos,cnt_neg

def main():
    shield_path = "pickles/shields/uncorr/*Berk*"
    stat_file_names_ppo = list(glob.glob(shield_path))

    shield_dict = dict()
    for file_name in stat_file_names_ppo:
        nice_name = to_nice_name(file_name)
        shield = load_pickle(file_name,exact_match=True)
        algo = "ppo" if "ppo" in file_name else "dqn"
        print(algo,file_name)
        shield_dict[(algo,nice_name)] = shield


    rules = defaultdict(list)
    for exp_name in shield_dict.keys():
        print(exp_name)
        print(len(shield_dict[exp_name].pos_rules_list))
        print(len(shield_dict[exp_name].neg_rules_list))
        for pos_rule_str in shield_dict[exp_name].pos_rules_list:
            rules[exp_name].append(string_to_rule(pos_rule_str))
        for neg_rule_str in shield_dict[exp_name].neg_rules_list:
            rules[exp_name].append(string_to_rule(neg_rule_str))

    table_stats = dict()
    for exp_name in rules.keys():
        rules_for_exp = rules[exp_name]

        nr_capsule_mentions_pos, nr_capsule_mentions_neg = cnt_mention(rules_for_exp,"capsules", exp_name)
        nr_capsule_mentions = nr_capsule_mentions_pos + nr_capsule_mentions_neg
        nr_food_mentions_pos, nr_food_mentions_neg = cnt_mention(rules_for_exp,"food", exp_name)
        nr_food_mentions = nr_food_mentions_pos + nr_food_mentions_neg
        nr_ghost_mentions_pos, nr_ghost_mentions_neg = cnt_mention(rules_for_exp,"ghosts", exp_name)
        nr_ghost_mentions = nr_ghost_mentions_pos + nr_ghost_mentions_neg

        nr_scared_mentions_pos, nr_scared_mentions_neg = cnt_mention(rules_for_exp,"ghosts-scared", exp_name)
        nr_scared_mentions = nr_scared_mentions_pos + nr_scared_mentions_neg
        nr_position_mentions_pos, nr_position_mentions_neg = cnt_mention(rules_for_exp,"position", exp_name)
        nr_position_mentions = nr_position_mentions_pos + nr_position_mentions_neg

        nr_rules =len(rules_for_exp)
        nr_pos_rules = len([r for r in rules_for_exp if r.polarity])
        nr_neg_rules = len([r for r in rules_for_exp if not r.polarity])
        print("**"*40)
        print(exp_name)
        print("Positive: ")
        print(f"Capsules: {nr_capsule_mentions_pos/nr_pos_rules}")
        print(f"Food: {nr_food_mentions_pos/nr_pos_rules}")
        print(f"Ghost: {nr_ghost_mentions_pos/nr_pos_rules}")
        print(f"Ghost Scared: {nr_scared_mentions_pos/nr_pos_rules}")
        print(f"Position: {nr_position_mentions_pos/nr_pos_rules}")

        print("Negative:")
        print(f"Capsules: {nr_capsule_mentions_neg/nr_neg_rules}")
        print(f"Food: {nr_food_mentions_neg/nr_neg_rules}")
        print(f"Ghost: {nr_ghost_mentions_neg/nr_neg_rules}")
        print(f"Ghost Scared: {nr_scared_mentions_neg/nr_neg_rules}")
        print(f"Position: {nr_position_mentions_neg/nr_neg_rules}")


        print("All:")
        print(f"Capsules: {nr_capsule_mentions/nr_rules}")
        print(f"Food: {nr_food_mentions/nr_rules}")
        print(f"Ghost: {nr_ghost_mentions/nr_rules}")
        print(f"Ghost Scared: {nr_scared_mentions/nr_rules}")
        print(f"Position: {nr_position_mentions/nr_rules}")
        table_stats[exp_name] = ("{0:.2f}".format(nr_capsule_mentions/nr_rules),
                                 "{0:.2f}".format(nr_ghost_mentions/nr_rules),
                                 "{0:.2f}".format(nr_scared_mentions/nr_rules))

    relevant_exp_names = ["small","medium","original"]

    table_string = """
        \\begin{tabular}{l|l|l|l|l}
        \\multirow{2}{*}{Experiment} & \\multicolumn{2}{|c|}{DQN} & \\multicolumn{2}{|c}{PPO} \\\\ \\cline{2-5}
         &  Ghosts & Scared &  Ghosts & Scared \\\\ \\hline 
         """

    for exp in relevant_exp_names:
        table_string += (f'{exp}'
                         f'& {table_stats[("dqn",exp)][1]} & {table_stats[("dqn",exp)][2]}  '
                         f'& {table_stats[("ppo",exp)][1]} & {table_stats[("ppo",exp)][2]}  '
                         f'\\\\ \\hline \n')
        print(exp)
        print("--"*20)
        print(table_stats[("ppo",exp)])
        print(table_stats[("dqn",exp)])

    table_string+="""
    \\end{tabular}
    """
    print(table_string)
if __name__ == "__main__":
    main()