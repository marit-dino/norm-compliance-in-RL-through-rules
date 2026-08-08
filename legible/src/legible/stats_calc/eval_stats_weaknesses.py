import glob
import statistics
from collections import defaultdict

import scipy.stats

from rule_learning.util import load_pickle
from evaluate_policy import EvalStats

def get_exp_name(eval_stats_path, file_name):

    pacman = "Berkeley" in file_name
    base_exp_name = file_name.replace(eval_stats_path, "")

    if pacman:
        base_exp_name = base_exp_name[:-len("_extended-8_1.pkl")]
    else:
        base_exp_name = base_exp_name[:-len("_1.pkl")]
    index = int(file_name[-len("1.pkl"):].replace(".pkl",""))

    return (base_exp_name.replace("ext_dqn","dqn").replace("_2500000_to_2500000_","_2500000_")
            .replace("_500000_to_500000","_500000").replace("_5000000_to_5000000_","_5000000_"),index)


def to_nice_name(exp_name):
    pacman = "Berkeley" in exp_name
    if pacman:
        level = exp_name[exp_name.index("level") + len("level_"):]
        level = level.replace("Classic","")
        level = level.replace("_no_capsules", "-nc")
    else:
        level = exp_name[len("dqn_"):exp_name.index("-v0")]
    return level

def filter_files(file_name_list,steps_list):
    filtered = []
    for steps in steps_list:
        filtered.extend(filter(lambda name : f"_{steps}_" in name,file_name_list))
    return filtered

def main():
    # random_path_part = "random_rules" if random_rules else "random/3"
    eval_stats_path_basic = "../pickles/eval_stats/"
    eval_stats_path_comb = "../pickles/eval_stats/combination/"
    eval_stats_path_random = f"../pickles/eval_stats/random/3/"
    eval_stats_path_random_rules = f"../pickles/eval_stats/random_rules/"
    eval_stats_path_ext = "../pickles/eval_stats/extended/"
    nr_step_hw = 500_000
    # file_names_basic = list(glob.glob(eval_stats_path_basic + "*.pkl"))
    # file_names_comb = list(glob.glob(eval_stats_path_comb + "*.pkl"))
    pacman = False
    do_weaknesses = False
    if pacman:
        state_file_names = list(glob.glob(eval_stats_path_basic +  "dqn_Berkeley*"))
        state_comb_file_names = list(glob.glob(eval_stats_path_comb + "dqn_Berkeley*"))
        state_ext_file_names = list(glob.glob(eval_stats_path_ext + "ext_dqn_Berkeley*"))
        state_random_file_names = list(glob.glob(eval_stats_path_random + "dqn_Berkeley*"))
        state_random_rule_file_names = list(glob.glob(eval_stats_path_random_rules + "dqn_Berkeley*"))
        state_file_names = filter_files(state_file_names,[2_500_000,5_000_000])
        state_comb_file_names = filter_files(state_comb_file_names,[2_500_000,5_000_000])
        state_random_file_names = filter_files(state_random_file_names,[2_500_000,5_000_000])
        state_random_rule_file_names = filter_files(state_random_rule_file_names,[2_500_000,5_000_000])
        state_ext_file_names = filter_files(state_ext_file_names,[f"{2_500_000}_to_{2_500_000}",f"{5_000_000}_to_{5_000_000}"])
    else:
        state_file_names = list(glob.glob(eval_stats_path_basic +  "*"))
        state_comb_file_names = list(glob.glob(eval_stats_path_comb + "*"))
        state_ext_file_names = list(glob.glob(eval_stats_path_ext + "*"))
        state_random_file_names = list(glob.glob(eval_stats_path_random + "*"))
        state_random_rule_file_names = list(glob.glob(eval_stats_path_random_rules + "*"))

        state_file_names_pac = list(glob.glob(eval_stats_path_basic +  "dqn_Berkeley*"))
        state_comb_file_names_pac = list(glob.glob(eval_stats_path_comb + "dqn_Berkeley*"))
        state_ext_file_names_pac = list(glob.glob(eval_stats_path_ext + "ext_dqn_Berkeley*"))
        state_random_file_names_pac = list(glob.glob(eval_stats_path_random + "dqn_Berkeley*"))
        state_random_rule_file_names_pac = list(glob.glob(eval_stats_path_random_rules + "dqn_Berkeley*"))
        for file_name in state_file_names_pac:
            state_file_names.remove(file_name)
        for file_name in state_comb_file_names_pac:
            state_comb_file_names.remove(file_name)
        for file_name in state_ext_file_names_pac:
            state_ext_file_names.remove(file_name)
        for file_name in state_random_file_names_pac:
            state_random_file_names.remove(file_name)
        for file_name in state_random_rule_file_names_pac:
            state_random_rule_file_names.remove(file_name)

        state_file_names = filter_files(state_file_names,[nr_step_hw])
        state_comb_file_names = filter_files(state_comb_file_names,[nr_step_hw])
        state_random_file_names = filter_files(state_random_file_names,[nr_step_hw])
        state_random_rule_file_names = filter_files(state_random_rule_file_names,[nr_step_hw])
        state_ext_file_names = filter_files(state_ext_file_names,[f"{nr_step_hw}_to_{nr_step_hw}"])

    print(state_random_rule_file_names)

    exp_names = set()
    for file_name in state_file_names:
        exp_name,_ = get_exp_name(eval_stats_path_basic, file_name)
        print(exp_name)
        exp_names.add(exp_name)


    results = defaultdict(list)
    for file_name in state_file_names:
        stats = load_pickle(file_name,exact_match=True)
        base_exp_name,index = get_exp_name(eval_stats_path_basic, file_name)
        results[base_exp_name].append((index,stats))

    random_results = defaultdict(list)
    for file_name in state_random_file_names:
        stats = load_pickle(file_name,exact_match=True)
        base_exp_name,index = get_exp_name(eval_stats_path_random, file_name)
        random_results[base_exp_name].append((index,stats))

    random_rule_results = defaultdict(list)
    for file_name in state_random_rule_file_names:
        stats = load_pickle(file_name,exact_match=True)
        base_exp_name,index = get_exp_name(eval_stats_path_random_rules, file_name)
        random_rule_results[base_exp_name].append((index,stats))

    results_comb = defaultdict(list)
    for file_name in state_comb_file_names:
        stats = load_pickle(file_name,exact_match=True)
        base_exp_name,index = get_exp_name(eval_stats_path_comb, file_name)
        print(base_exp_name,stats[1].avg_rew)
        # print(to_nice_name(base_exp_name))
        # nice_exp_name = to_nice_name(base_exp_name)
        results_comb[base_exp_name].append((index,stats))

    results_ext = defaultdict(list)
    for file_name in state_ext_file_names:
        stats = load_pickle(file_name,exact_match=True)
        base_exp_name,index = get_exp_name(eval_stats_path_ext, file_name)
        print(base_exp_name,stats.avg_rew)
        # print(to_nice_name(base_exp_name))
        # nice_exp_name = to_nice_name(base_exp_name)
        results_ext[base_exp_name].append((index,stats))

    def calc_stats_from_list(value_list):
        if len(value_list) == 0:
            return 0,0,0
        return min(value_list),statistics.mean(value_list),max(value_list)

    if pacman:
        exp_name_list = ["small", "small-nc", "medium", "medium-nc", "original", "original-nc"]

    else:
        exp_name_list = ["highway", "highway-fast", "merge", "intersection", "roundabout"]
        # exp_name_list = ["highway-fast", "merge", "intersection", "roundabout"]

    if do_weaknesses:
        improved_counts = do_improved_tests(results, False)
        print("**"*20)
        improved_random_counts = do_improved_tests(random_results,True)
        improved_random_rule_counts = do_improved_tests(random_rule_results,False)

        table_stats_improved_cnt = dict()
        table_stats_improved_ratio = dict()
        table_stats_large_improved_ratio = dict()
        table_stats_test_improved_ratio = dict()
        table_stats_test_improved_random_ratio = dict()
        table_stats_test_improved_random_rule_ratio = dict()
        table_stats_eval_cnt = dict()
        for exp_name in improved_counts.keys():
            improved_ratios_single = []
            improved_counts_single = []
            improved_large_ratio_single = []
            improved_test_ratio_single = []
            improved_test_random_ratio_single = []
            improved_test_random_rule_ratio_single = []
            all_evals_counts_single = []
            for (exp_index,n_improved,n_large_improved,n_test_improved,n_all) in improved_counts[exp_name]:
                improved_ratios_single.append(n_improved/n_all)
                improved_counts_single.append(n_improved)
                all_evals_counts_single.append(n_all)
                improved_test_ratio_single.append(n_test_improved/n_all)
                improved_large_ratio_single.append(n_large_improved/n_all)

            for (exp_index,n_improved,n_large_improved,n_test_improved,n_all) in improved_random_counts[exp_name]:
                improved_test_random_ratio_single.append(n_test_improved / n_all)
            for (exp_index,n_improved,n_large_improved,n_test_improved,n_all) in improved_random_rule_counts[exp_name]:
                improved_test_random_rule_ratio_single.append(n_test_improved / n_all)

            print(f"Changing {exp_name} to")
            exp_name = to_nice_name(exp_name)
            print(exp_name)
            table_stats_improved_ratio[exp_name] = calc_stats_from_list(improved_ratios_single)
            table_stats_improved_cnt[exp_name] = calc_stats_from_list(improved_counts_single)
            table_stats_eval_cnt[exp_name] = calc_stats_from_list(all_evals_counts_single)
            table_stats_large_improved_ratio[exp_name] = calc_stats_from_list(improved_large_ratio_single)
            table_stats_test_improved_ratio[exp_name] = calc_stats_from_list(improved_test_ratio_single)
            table_stats_test_improved_random_ratio[exp_name] = calc_stats_from_list(improved_test_random_ratio_single)
            table_stats_test_improved_random_rule_ratio[exp_name] = calc_stats_from_list(improved_test_random_rule_ratio_single)

            print(f"Improved count: {table_stats_improved_cnt[exp_name]}")
            print(f"Eval count: {table_stats_eval_cnt[exp_name]}")
            print(f"Improved ratios: {table_stats_improved_ratio[exp_name]}")
            print(f"Large Improved ratios: {table_stats_large_improved_ratio[exp_name]}")
            print(f"Test Improved ratios: {table_stats_test_improved_ratio[exp_name]}")
            print(f"Test Improved random ratios: {table_stats_test_improved_random_ratio[exp_name]}")
            print(f"Test Improved random rule ratios: {table_stats_test_improved_random_rule_ratio[exp_name]}")

        # table_string = """
        #          \\begin{tabular}{l|c|c|c|c|c|c}
        #          \\multirow{2}{*}{Experiment} &  \\multicolumn{3}{|c|}{Detected Weaknesses} & \\multicolumn{3}{|c}{\\# Evaluations} \\\\ \\cline{2-7}
        #           & min. & mean & max. & min. & mean & max. \\\\ \\hline
        #           """
        table_string = """
         \\begin{tabular}{l|c|c|c|c}
         \\multirow{2}{*}{Experiment} &  \\multicolumn{3}{|c|}{Detected Weaknesses} &  \\ \\multirow{2}{*}{# Evaluations} \\\\ \\cline{2-4}
          & RT & RR & MT &  \\\\ \\hline 
          """

        for exp_name in exp_name_list:
            min_ratio = float_to_round_string(table_stats_improved_ratio[exp_name][0])
            avg_ratio = float_to_round_string(table_stats_improved_ratio[exp_name][1])
            max_ratio = float_to_round_string(table_stats_improved_ratio[exp_name][2])

            min_large_ratio = float_to_round_string(table_stats_large_improved_ratio[exp_name][0])
            avg_large_ratio = float_to_round_string(table_stats_large_improved_ratio[exp_name][1])
            max_large_ratio = float_to_round_string(table_stats_large_improved_ratio[exp_name][2])

            min_test_ratio = float_to_round_string(table_stats_test_improved_ratio[exp_name][0])
            avg_test_ratio = float_to_round_string(table_stats_test_improved_ratio[exp_name][1])
            max_test_ratio = float_to_round_string(table_stats_test_improved_ratio[exp_name][2])

            avg_test_random_ratio = float_to_round_string(table_stats_test_improved_random_ratio[exp_name][1])
            avg_test_random_rule_ratio = float_to_round_string(table_stats_test_improved_random_rule_ratio[exp_name][1])

            min_cnt = float_to_round_string(table_stats_eval_cnt[exp_name][0],decimal_places=1)
            avg_cnt = float_to_round_string(table_stats_eval_cnt[exp_name][1],decimal_places=1)
            max_cnt = float_to_round_string(table_stats_eval_cnt[exp_name][2],decimal_places=1)
            # table_string += (f" {exp_name} & ${min_ratio} \\mid {min_large_ratio}$"
            #                  f"& ${avg_ratio} \\mid {avg_large_ratio}$"
            #                  f"& ${max_ratio} \\mid {max_large_ratio}$ & "
            #                  f"{min_cnt}& {avg_cnt}& {max_cnt} \\\\ \\hline \n")
            # table_string += (f" {exp_name} & \\makecell{{${min_ratio}$ \\\\ ${min_large_ratio}$ }}"
            #                  f"& \\makecell{{ ${avg_ratio}$ \\\\ ${avg_large_ratio}$ }}"
            #                  f"& \\makecell{{ ${max_ratio}$ \\\\ ${max_large_ratio}$ }} & "
            #                  f"${min_cnt}$ & ${avg_cnt}$ & ${max_cnt}$\\\\ \\hline \n")

            # table_string += (f" {exp_name} & ${min_test_ratio}$ "
            #                  f"& ${avg_test_ratio}$ "
            #                  f"& ${max_test_ratio}$  & "
            #                  f"${min_cnt}$ & ${avg_cnt}$ & ${max_cnt}$\\\\ \\hline \n")

            table_string += (f" {exp_name} "
                             f"& ${avg_test_random_ratio}$ "
                             f"& ${avg_test_random_rule_ratio}$ "
                             f"& ${avg_test_ratio}$ "
                             f"& ${avg_cnt}$ \\\\ \\hline \n")
        table_string += """
         \\end{tabular}
         """
        print(table_string)
    else:

        table_stats_comb_rew = dict()
        table_stats_base_rew = dict()
        table_stats_comb_stderr = dict()
        table_stats_base_stderr = dict()
        table_stats_ext_rew = dict()
        table_stats_ext_stderr = dict()
        table_stats_comb_rules = dict()
        for exp_name in results.keys():
            avg_rews_comb = []
            stderrs_comb = []
            avg_rews_base = []
            stderrs_base = []
            avg_rews_ext = []
            stderrs_ext = []
            rule_sets = []
            for (exp_index,(rules_list, single_results)) in results_comb[exp_name]:
                avg_rews_comb.append(single_results.avg_rew)
                stderrs_comb.append(single_results.stderr_rew)
                rule_sets.append(len(rules_list))

            for (exp_index,single_results) in results_ext[exp_name]:
                avg_rews_ext.append(single_results.avg_rew)
                stderrs_ext.append(single_results.stderr_rew)

            for (exp_index, single_results) in results[exp_name]:
                avg_rews_base.append(single_results[-1].avg_rew)
                stderrs_base.append(single_results[-1].stderr_rew)
            print(f"Changing {exp_name} to")
            exp_name = to_nice_name(exp_name)
            print(exp_name)
            table_stats_base_rew[exp_name] = calc_stats_from_list(avg_rews_base)
            table_stats_base_stderr[exp_name] = calc_stats_from_list(stderrs_base)
            table_stats_comb_rules[exp_name] = calc_stats_from_list(rule_sets)
            table_stats_comb_rew[exp_name] = calc_stats_from_list(avg_rews_comb)
            table_stats_comb_stderr[exp_name] = calc_stats_from_list(stderrs_comb)

            table_stats_ext_rew[exp_name] = calc_stats_from_list(avg_rews_ext)
            table_stats_ext_stderr[exp_name] = calc_stats_from_list(stderrs_ext)
            print(f"Base: {table_stats_base_rew[exp_name]} +- {table_stats_base_stderr[exp_name]}")
            print(f"Number of rule sets: {table_stats_comb_rules[exp_name]}")
            print(f"Improved: {table_stats_comb_rew[exp_name]} +- {table_stats_comb_stderr[exp_name]}")
            print(f"Extended: {table_stats_ext_rew[exp_name]} +- {table_stats_ext_stderr[exp_name]}")

        table_string = """
                 \\begin{tabular}{l|c|c|c|c}
                 Experiment &  Base & Rule-Guided & Ext. Training & RS  \\\\ \\hline
                  """

        for exp_name in exp_name_list:
            print(table_stats_base_rew.keys())
            dec_places=1
            base_rew_str = (f"\\makecell{{${float_to_round_string(table_stats_base_rew[exp_name][1],decimal_places=dec_places)}$ \\\\ $\\pm "
                            f"{float_to_round_string(table_stats_base_stderr[exp_name][1],decimal_places=dec_places)}$}}")
            improved_rew_str = (f"\\makecell{{ ${float_to_round_string(table_stats_comb_rew[exp_name][1],decimal_places=dec_places)}$ \\\\ $\\pm "
                            f"{float_to_round_string(table_stats_comb_stderr[exp_name][1],decimal_places=dec_places)}$}}")
            ext_rew_str = (f"\\makecell{{ ${float_to_round_string(table_stats_ext_rew[exp_name][1],decimal_places=dec_places)}$ \\\\ $\\pm "
                            f"{float_to_round_string(table_stats_ext_stderr[exp_name][1],decimal_places=dec_places)}$ }}")
            rules_used = float_to_round_string(table_stats_comb_rules[exp_name][1],decimal_places=dec_places)

            table_string += f" {exp_name} & {base_rew_str} & {improved_rew_str} & {ext_rew_str} & ${rules_used}$ \\\\ \\hline \n"
        table_string += """
                 \\end{tabular}
                 """
        print(table_string)


def do_improved_tests(results, random_eval, p_threshold = 0.05):
    improved_counts = defaultdict(list)
    for base_exp_name in results.keys():
        for (exp_index, single_results) in results[base_exp_name]:
            n_improved = 0
            n_large_improved = 0
            n_test_improved = 0

            unguided_stats = single_results[-1]
            for result_index, other_stats in single_results.items():
                if random_eval and result_index != -1:
                    other_stats = other_stats[1]
                if result_index != -1:
                    if other_stats.avg_rew > unguided_stats.avg_rew:
                        n_improved += 1
                    if other_stats.avg_rew - other_stats.stderr_rew > unguided_stats.avg_rew + unguided_stats.stderr_rew:
                        n_large_improved += 1
                    unguided_rews = unguided_stats.cum_rews
                    other_rews = other_stats.cum_rews
                    test_res = scipy.stats.ttest_ind(unguided_rews, other_rews, equal_var=False, alternative="less")
                    p_val = test_res.pvalue
                    print(statistics.mean(unguided_rews), statistics.mean(other_rews), p_val)
                    if p_val < p_threshold:
                        n_test_improved += 1
            n_all = len(single_results) - 1  # subtract base stats
            print(f"Exp {base_exp_name},{exp_index} {n_test_improved}, {n_all}")
            improved_counts[base_exp_name].append((exp_index, n_improved, n_large_improved, n_test_improved, n_all))
    return improved_counts


def float_to_round_string(float_val, decimal_places = 2):
    return ("{0:." + str(decimal_places) + "f}").format(float_val)



if __name__ == "__main__":
    main()