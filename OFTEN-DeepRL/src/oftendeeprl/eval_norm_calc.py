import glob
import os

from util import load_pickle
from evaluate_policy import EvalStats

env = "BerkeleyPacman-v0"
steps = 1000000
feat_ext = "complete"
field_name = "agent_eaten"

def load_files(path_prefix):
    stats_list = []
    for file in glob.glob(path_prefix + "*"):
        eval_stat = load_pickle(file,exact_match=True)
        stats_list.append(eval_stat)
    return stats_list


def stats_avgs(stats : list[EvalStats],last, base=False):
    wins = 0
    rewards = 0
    eaten_blue = 0
    eaten_green = 0
    avg_step_time = 0
    print(stats)
    avg_step_time_n = 0
    for stat in stats:
        wins += stat.nr_wins / len(stat.additional_data)
        rewards += stat.avg_rew
        eaten_blue_single = 0
        eaten_green_single = 0
        try:
            avg_step_time += stat.avg_step_times
            avg_step_time_n += 1
        except:
            pass
        for d in stat.additional_data:
            agent_eaten_single = d[field_name]
            eaten_blue_single += agent_eaten_single[1]
            for i in range(2,len(agent_eaten_single)):
                eaten_green_single += agent_eaten_single[i]
        eaten_blue_single /= len(stat.additional_data)
        eaten_green_single /= len(stat.additional_data)
        eaten_blue += eaten_blue_single
        eaten_green += eaten_green_single

    n_stats = len(stats)
    if base:
        s = "\\multirow{2}{*}{"
        e = "}"
    else:
        s = ""
        e = ""
    # return (f"{s}%.2f{e} "
    #         f"& {s}%.2f{e} "
    #         f"& {s}%.2f / %.2f{e} "
    #         f"& {s}%.2f{e} & ")%((wins/n_stats),(rewards/n_stats),(eaten_blue/n_stats),eaten_green/n_stats,
    #                                     avg_step_time / avg_step_time_n)
    last_cr = "" if last else "&"
    return (f"{s}%.2f{e} "
            f"& {s}%.2f{e} "
            f"& {s}%.2f / %.2f{e} {last_cr}")%((wins/n_stats),(rewards/n_stats),(eaten_blue/n_stats),eaten_green/n_stats)
    # return f"{wins/n_stats} & {rewards/n_stats} & {eaten_blue/n_stats} / {eaten_green/n_stats}"

run = True
if run:
    table_string = """
        \\begin{tabular}{l|l|l|l|l|l|l|l|l|l}
        \\multirow{2}{*}{Experiment} & \\multicolumn{3}{|c|}{Base} & 
        \\multicolumn{3}{|c|}{\\torres} & \\multicolumn{3}{|c}{Policy Fixes} \\\\ \\cline{2-10}
         & SR & Rew. & NV & SR & Rew. & NV & SR & Rew. & NV \\\\ \\hline 
         """
    for level in [ "smallClassic", "mediumClassic", "originalClassic"]:
        for vegetarian in [True,False]:
            n_ghosts = 4 if "original" in level else 2
            print(f"Evaluation results: for {level}")
            print("**"*10)
            veg_string = "vegetarian" if vegetarian else "vegan"
            norm_descriptor = f"_{veg_string}_2_5_"
            eval_path_base = f"pickles/eval_stats/base/dqn_{env}_{steps}_level_{level}_{feat_ext}" #_1.pkl"
            eval_path_norm = (f"pickles/eval_stats/extended/margin50/norm_guided_dqn_{norm_descriptor}_{env}_"
                              f"{steps}_to_{steps}_level_{level}_{feat_ext}") #_1.pkl")
            eval_path_fix = (f"pickles/eval_stats/policy_fix/dqn{norm_descriptor}{env}_"
                              f"{steps}_level_{level}_{feat_ext}") #_1.pkl")
            print(eval_path_base)
            print(eval_path_norm)
            print(eval_path_fix)
            stats_base = load_files(eval_path_base)
            stats_norm = load_files(eval_path_norm)
            stats_fix = load_files(eval_path_fix)

            table_line = f"{level.replace('Classic','')}-"
            table_line += "v" if vegetarian else "v+"
            table_line += " & "
            if vegetarian:
                table_line += stats_avgs(stats_base,last=False,base=True)
            else:
                table_line += " & & &"
            table_line += stats_avgs(stats_fix,last=False)
            table_line += stats_avgs(stats_norm,last=True)
            line_rule = "\\cline{5-10}" if vegetarian else "\\hline"
            table_string += table_line + f"\\\\ {line_rule}" + os.linesep
    table_string+= "\\end{tabular}" + os.linesep
    print(table_string)

