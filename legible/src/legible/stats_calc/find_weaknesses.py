import glob

from rule_learning.util import load_pickle
from shield.rule_classes import string_to_rule, Rule, RuleBody
from shield.shields import RuleChooser


def get_short_exp_name_from_stat(file_name):
    level = ""
    if "medium" in file_name:
        level = "medium"
    elif "small" in file_name:
        level = "small"
    elif "original" in file_name:
        level = "original"
    mode = "NC" if "no_capsules" in file_name else ""
    return level,mode


def removed_ignored_features(all_rules_parsed, level):
    from create_rules_pacman import remove_feat_4_ghosts,remove_feat_2_ghosts
    if level == "original":
        remove_list = remove_feat_4_ghosts
    else:
        remove_list = remove_feat_2_ghosts

    def remove_ignored_from_rules(rule : Rule):
        new_conds = [cond for cond in list(rule.rule_body.conditions) if cond.feature not in remove_list]
        return Rule(rule.polarity,rule.rule_head,RuleBody(new_conds))
    return [remove_ignored_from_rules(r) for r in all_rules_parsed]


def stronger_than(r, r_reference):
    if r.polarity == r_reference.polarity and r.rule_head == r_reference.rule_head:
        conds_r = r.rule_body.conditions
        conds_ref = r_reference.rule_body.conditions
        for cond_ref in conds_ref:
            if cond_ref not in conds_r:
                return False
        return len(conds_r) > len(conds_ref)
    else:
        return False


def stronger_rule_present(r_reference, all_rules_parsed):
    for r in all_rules_parsed:
        if stronger_than(r, r_reference):
            return True,r
    return False, None


def main():

    eval_stats_path_basic = "../pickles/eval_stats/"
    stats_file_names_basic = list(glob.glob(eval_stats_path_basic + "*.pkl"))
    exp_data = dict()
    for file_name in stats_file_names_basic:
        level,mode = get_short_exp_name_from_stat(file_name)
        stats = load_pickle(file_name,exact_match=True)
        sorted_stats = sorted(list(stats.items()),key=lambda x : x[1].avg_rew,reverse=True)
        mode_long = "no_capsules_" if "NC" in mode else ""
        feat_step_string = "117_5000000" if "original" in level  else "69_2500000"
        shield_name = f"dqn_BerkeleyPacman-v0_{level}Classic_{mode_long}feat_{feat_step_string}_shield_1.pkl"
        shield = load_pickle(f"pickles/shields/improved/{shield_name}",exact_match=True)
        exp_data[(level,mode)] = (sorted_stats,shield)

    print("Highest reward achieved:")
    levels = ["small","medium","original"]
    modes = ["NC", ""]
    for level in levels:
        for mode in modes:
            stats = exp_data[(level,mode)][0]

            shield = exp_data[(level,mode)][1]
            best_rule_index = stats[0][0]

            rule_chooser = RuleChooser(shield)
            print("--"*20)
            print(f"Experiment: {level} {mode}")
            print(f"Best rule index: {best_rule_index}")
            gen_rules, base_rule = rule_chooser.select_rules([best_rule_index],return_keys=True)
            print(f"Base: '{base_rule}' generalized to:")
            all_rules_parsed = [string_to_rule(r) for r in shield.neg_rules_list] + [string_to_rule(r) for r in shield.pos_rules_list]
            all_rules_parsed = removed_ignored_features(all_rules_parsed,level)
            for r in gen_rules:
                r_as_rule = string_to_rule(r)
                print(r)
                print(f"Contained in original rules: {r_as_rule in all_rules_parsed}")
                has_stronger,stronger_rule = stronger_rule_present(r_as_rule, all_rules_parsed)
                if has_stronger:
                    print(f"Original rules contain stronger version: {has_stronger} with {stronger_rule}")
                else:
                    print(f"Original rules contain stronger version: {has_stronger}")

if __name__ == "__main__":
    main()