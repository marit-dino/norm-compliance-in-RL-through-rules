import glob
import statistics
from collections import defaultdict

from rule_learning.util import load_pickle
from evaluate_policy import EvalStats


def to_nice_name(exp_name):
    mode = "rule-guided" if exp_name.startswith("shielded") else "normal"
    level = exp_name[exp_name.index("level") + len("level_"):]
    level = level.replace("Classic","")
    level = level.replace("_no_capsules", "(NC)")
    return mode,level


def main():
    eval_stats_path_basic = "../pickles/eval_stats/"
    eval_stats_path = "../pickles/eval_stats/extended/"
    eval_comb_stats_path = "../pickles/eval_stats/combination/"
    state_file_names = list(glob.glob(eval_stats_path + "*"))
    state_comb_file_names = list(glob.glob(eval_comb_stats_path + "*"))
    stats_file_names_basic = list(glob.glob(eval_stats_path_basic + "*.pkl"))
    exp_names = set()
    for file_name in state_file_names:
        base_exp_name = get_exp_name(eval_stats_path, file_name)
        print(base_exp_name)
        exp_names.add(base_exp_name)
    comp_exp_names = set()
    for file_name in state_comb_file_names:
        base_exp_name = get_exp_name(eval_stats_path, file_name)
        print(base_exp_name)
        comp_exp_names.add(base_exp_name)

    # for file_name in stats_file_names_basic:
    #     base_exp_name = get_exp_name(eval_stats_path, file_name)
    #     print(base_exp_name)
    #     comp_exp_names.add(base_exp_name)

    results = defaultdict(list)
    for file_name in state_file_names:
        stats = load_pickle(file_name,exact_match=True)
        base_exp_name = get_exp_name(eval_stats_path, file_name)
        results[base_exp_name].append((file_name,stats))

    results_basic = dict()
    for file_name in stats_file_names_basic:
        stats = load_pickle(file_name,exact_match=True)
        base_exp_name = get_exp_name(eval_stats_path, file_name)
        print(to_nice_name(base_exp_name))
        nice_exp_name = to_nice_name(base_exp_name)
        results_basic[nice_exp_name[1]] = stats

    # comb_results = defaultdict(list)
    # for file_name in state_comb_file_names:
    #     stats = load_pickle(file_name,exact_match=True)
    #     base_exp_name = get_exp_name(eval_stats_path, file_name)
    #     comb_results[base_exp_name].append((file_name,stats))
    sorted_basic_results = sorted(results_basic[("small")].items(),key=lambda x : x[1].avg_rew,reverse=True)
    print("FOOBAR")
    print(sorted_basic_results[0])

    table_stats_mean = dict()
    table_stats_max = dict()
    table_stats_min = dict()
    for exp_name in results.keys():
        nice_exp_name = to_nice_name(exp_name)

        print(nice_exp_name)
        avg_rews = []
        for (file_name,result) in results[exp_name]:
            avg_rews.append(result.avg_rew)
            # print(file_name)
            # print(result.avg_rew)
        mean = statistics.mean(avg_rews)
        print(f"Mean: {mean}")
        max_rew = max(avg_rews)
        print(f"Max: {max_rew}")
        min_rew = min(avg_rews)
        print(f"Min: {min_rew}")
        table_stats_mean[nice_exp_name] = mean
        table_stats_max[nice_exp_name] = max_rew
        table_stats_min[nice_exp_name] = min_rew

    exp_name_list = ["small","small(NC)","medium","medium(NC)","original","original(NC)"]
    table_string = """
    \\begin{tabular}{l|l|l|l|l|l|l|l}
    \\multirow{2}{*}{Experiment} & \\multirow{2}{*}{Base} & \\multicolumn{3}{|c|}{Rule-Guided} & \\multicolumn{3}{|c}{Extended} \\\\ \\cline{2-8}
     & min. & mean & max. & min. & mean & max. \\\\ \\hline """

    for exp_name in exp_name_list:
        min_rule = float_to_round_string(table_stats_min[("rule-guided",exp_name)])
        avg_rule = float_to_round_string(table_stats_mean[("rule-guided",exp_name)])
        max_rule = float_to_round_string(table_stats_max[("rule-guided",exp_name)])
        min_normal = float_to_round_string(table_stats_min[("normal",exp_name)])
        avg_normal = float_to_round_string(table_stats_mean[("normal",exp_name)])
        max_normal = float_to_round_string(table_stats_max[("normal",exp_name)])
        table_string += (f" {exp_name} & {float_to_round_string(results_basic[exp_name][-1].avg_rew)}"
                         f"& {min_rule}& {avg_rule}& {max_rule}& {min_normal}& {avg_normal}& {max_normal} \\\\ \\hline \n")
    table_string+="""
    \\end{tabular}
    """
    print(table_string)

def float_to_round_string(float_val, decimal_places = 2):
    return "{0:.2f}".format(float_val)

def get_exp_name(eval_stats_path, file_name):
    base_exp_name = file_name.replace(eval_stats_path, "")
    base_exp_name = base_exp_name[:-len("_extended-8_1.pkl")]
    return base_exp_name


if __name__ == "__main__":
    main()