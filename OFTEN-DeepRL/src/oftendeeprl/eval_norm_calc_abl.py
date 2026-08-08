import glob
import os

from eval_norm_calc import stats_avgs
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
table_string = """
    \\begin{tabular}{l|l|l|l}
    Config. & SR & Rew. & NV \\\\ \\hline
     """

level = "smallClassic"
vegetarian = False
n_ghosts = 4 if "original" in level else 2
print(f"Evaluation results: for {level}")
print("**"*10)
veg_string = "vegetarian" if vegetarian else "vegan"
norm_descriptor = f"_{veg_string}_2_5_"
eval_path_base = f"pickles/eval_stats/base/dqn_{env}_{steps}_level_{level}_{feat_ext}" #_1.pkl"
eval_path_norm = (f"pickles/eval_stats/extended/margin50/norm_guided_dqn_{norm_descriptor}_{env}_"
                  f"{steps}_to_{steps}_level_{level}_{feat_ext}") #_1.pkl")
eval_path_no_exp = (f"pickles/eval_stats/extended/margin50/norm_guided_dqn_no_exp_{norm_descriptor}_{env}_"
                  f"{steps}_to_{steps}_level_{level}_{feat_ext}") #_1.pkl")
eval_path_no_filter = (f"pickles/eval_stats/extended/margin50/norm_guided_dqn_no_filter_{norm_descriptor}_{env}_"
                  f"{steps}_to_{steps}_level_{level}_{feat_ext}") #_1.pkl")

eval_path_fresh = (f"pickles/eval_stats/extended/margin50/norm_guided_dqn_{norm_descriptor}_{env}_"
                  f"0_to_{steps}_level_{level}_{feat_ext}")

print(eval_path_base)
print(eval_path_norm)
print(eval_path_no_exp)
print(eval_path_no_filter)
print(eval_path_fresh)
stats_base = load_files(eval_path_base)
stats_norm = load_files(eval_path_norm)
stats_no_exp = load_files(eval_path_no_exp)
stats_no_filter = load_files(eval_path_no_filter)
stats_fresh = load_files(eval_path_fresh)

stats_list = [stats_base,stats_norm,stats_no_exp,stats_no_filter,stats_fresh]
config_names = ["Base", "\\torres", "No $J_{E}$", "No filtering", "No Pre-training"]

for stats,config_name in zip(stats_list,config_names):
    table_line = f"{config_name} & "
    table_line += stats_avgs(stats,last=True)
    table_string += table_line + f"\\\\ \\hline" + os.linesep
table_string+= "\\end{tabular}" + os.linesep
print(table_string)

        # eval_base = load_pickle(eval_path_base,exact_match=True)
        # eval_norm = load_pickle(eval_path_norm,exact_match=True)

        # agent_eaten_all = [0] * (n_ghosts + 1)
        # for d in eval_base.additional_data:
        #     agent_eaten_single = d[field_name]
        #     for i in range(len(agent_eaten_single)):
        #         agent_eaten_all[i] += agent_eaten_single[i]
        #
        #
        # print(f"Base eaten count: {agent_eaten_all[1:]}")
        # print(f"Base wins: {eval_base.nr_wins}")
        # print(f"Base reward: {eval_base.avg_rew}")
        # print(f"Base ep len: {eval_base.avg_steps}")
        #
        #
        # norm_agent_eaten_all = [0] * (n_ghosts + 1)
        # for d in eval_norm.additional_data:
        #     norm_agent_eaten_single = d[field_name]
        #     for i in range(len(norm_agent_eaten_single)):
        #         norm_agent_eaten_all[i] += norm_agent_eaten_single[i]
        #
        # print(f"Norm eaten count: {norm_agent_eaten_all[1:]}")
        # print(f"Norm wins: {eval_norm.nr_wins}")
        # print(f"Norm reward: {eval_norm.avg_rew}")
        # print(f"Norm ep len: {eval_norm.avg_steps}")