import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass

import gym_pacman
import torch

from sb3_ext.DQfD import RuleDQfD
from util import load_pickle, load_model, save_pickle
from env_util import create_environment_and_modelname, get_norm_helper, get_gym_to_action_names

from sumo_single_intersection import IncludeAmbulancesObservationFunction
from env_util import get_sumo_level_name

class EvalStats:
    def __init__(self,all_rews,action_changes_list,wins, additional_data, episode_times):
        self.all_rews = all_rews
        self.action_changes = action_changes_list
        self.cum_rews = []
        self.steps =[]
        nr_eps = len(self.all_rews)
        for rewards in all_rews:
            cum_rew = sum(rewards)
            nr_steps = len(rewards)
            self.cum_rews.append(cum_rew)
            self.steps.append(nr_steps)
        self.avg_rew = statistics.mean(self.cum_rews)
        self.avg_steps = statistics.mean(self.steps)
        self.avg_action_changes = statistics.mean(action_changes_list)
        self.stderr_rew = statistics.stdev(self.cum_rews) / math.sqrt(nr_eps)
        self.stderr_steps = statistics.stdev(self.steps) / math.sqrt(nr_eps)
        self.stderr_action_changes = statistics.stdev(action_changes_list) / math.sqrt(nr_eps)
        self.wins = wins
        self.nr_wins = sum(wins)
        self.additional_data = additional_data
        self.episode_times = episode_times
        self.avg_step_times = statistics.mean([e/s for s,e in zip(self.steps,self.episode_times)])

def create_rule_string_for_pos_neg(shield,shield_rule_nrs):
    rule_string = ''
    for shield_rule_nr in shield_rule_nrs:
        if shield_rule_nr < len(shield.pos_rules_list):
            rule_string += shield.pos_rules_list[shield_rule_nr] + "\n"
        else:
            rule_string += shield.neg_rules_list[shield_rule_nr - len(shield.pos_rules_list)] + "\n"
    return rule_string

def eval_single_eps(env, algo_name,model, action_tensor,norm_descriptor,gym_to_action_names, action_names_to_int,norm_helper):
    obs, info = env.reset()
    win = False
    rewards = []
    action_changes = 0
    violation = False
    if norm_helper is not None:
        if "Sumo" in str(env):
            env.reset()
            clingo_excluded = []
            clingo_static = norm_helper.get_clingo_static(env=env,
                                                        excluded=clingo_excluded)
            norm_helper.setup(clingo_static, clingo_excluded)
        elif "Gardener" in str(env):
            norm_helper.setup(env.get_wrapper_attr("instance"))
    while True:
        if "dqn" in algo_name or algo_name.startswith("dqn-n"):
            if norm_helper is not None:
                if "Sumo" in str(env):
                    env_state = {'env': env}
                elif "Gardener" in str(env):
                    env_state = env.get_wrapper_attr("instance")
                else:
                    env_state = env.get_state()
                fix_action, orig_action = RuleDQfD.policy_fix_single(margin = None, gym_to_action_names = gym_to_action_names,
                                                                     nr_actions = action_tensor.shape[0],
                                                                 policy = model.policy, norm_helper = norm_helper,
                                                                 chosen_actions = None, env_state = env_state, i = None, obs_i = obs)
                if fix_action == "Stop":
                    fix_action = random.randint(0, 3)
                else:
                    fix_action = action_names_to_int[fix_action]

                action = fix_action
                if orig_action != fix_action:
                    action_changes += 1
            else:
                action, _states = model.predict(obs, deterministic=False)
        else:
            raise Exception("Unsupported")

        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        if "Sumo" in str(env):
            max_time = IncludeAmbulancesObservationFunction.normalize_ambulance_wait_time(10)
            if any(obs[-4:]>max_time):
                violation = True
            # otherwise, remain with the same value
            info["ambulance_waited_too_long"] = violation

        if term and reward > 0:
            win = True # TODO check if true for all environments
        if term or trunc:
            break

    return win, rewards, action_changes, info

def evaluate(env,algo_name, model,nr_eps,action_tensor,additional_data_labels,norm_descriptor, norm_helper):
    action_changes_list = []

    print(f"Evaluation:" )
    all_rews = []
    wins = []
    additional_data = []
    ep_times = []

    gym_to_action_names = get_gym_to_action_names(env_name)
    action_names_to_int = {a : i for i,a in gym_to_action_names.items()}
    for i in range(nr_eps):
        additional_data_single = dict()
        start_time = time.time()
        win, rewards,action_changes, last_info = eval_single_eps(env,algo_name,model,action_tensor,norm_descriptor,
                                                                 gym_to_action_names,action_names_to_int,norm_helper)
        end_time = time.time()
        ep_times.append(end_time - start_time)
        wins.append(win)
        all_rews.append(rewards)
        action_changes_list.append(action_changes)
        for l in additional_data_labels:
            additional_data_single[l] = last_info[l]
        additional_data.append(additional_data_single)

    eval_stats = EvalStats(all_rews,action_changes_list,wins,additional_data,ep_times)
    print(f"Nr. wins: {eval_stats.nr_wins}")
    print(f"Avg. reward: {eval_stats.avg_rew} with SE {eval_stats.stderr_rew}")
    print(f"Avg. steps: {eval_stats.avg_steps} with SE {eval_stats.stderr_steps}")
    print(f"Avg. action changes: {eval_stats.avg_action_changes} with SE {eval_stats.stderr_action_changes}")
    if "gardener" in env_name:
        killed_plants = [x["nr-killed-plants"] for x in eval_stats.additional_data]
        killed_frogs = [x["nr-killed-frogs"] for x in eval_stats.additional_data]

        nr_eps = len(killed_plants)

        mean_killed_frogs = statistics.mean(killed_frogs)
        std_error_killed_frogs = statistics.stdev(killed_frogs) / math.sqrt(nr_eps)
        mean_killed_plants = statistics.mean(killed_plants)
        std_error_killed_plants = statistics.stdev(killed_plants) / math.sqrt(nr_eps)

        print(f"Avg. killed frogs: {mean_killed_frogs} with SE {std_error_killed_frogs}")
        print(f"Abg. killed plants: {mean_killed_plants} with SE {std_error_killed_plants}")
    elif "sumo" in env_name:
        nr_violations = sum([1 for x in eval_stats.additional_data if x['ambulance_waited_too_long']])
        print(f"Nr. violations: {nr_violations}")
    return eval_stats

def complete_policy_eval(algo_name, env_name, mode, steps, nr_eps,
                         feature_extractor,ext_to,
                         exact_model_number,additional_data_labels,norm_descriptor, norm_helper):

    env, model_name, model_path = create_environment_and_modelname(algo_name, env_name, mode, steps, feature_extractor=feature_extractor)
    extended = algo_name == "ext_dqn"
    norm_guided = "norm_guided_dqn" in algo_name
    if ext_to is not None:
        if "Berkeley" in env_name:
            model_name = model_name.replace(f"{steps}_level",f"{steps}_to_{ext_to}_level")
            model_path = model_path.replace(f"{steps}_level",f"{steps}_to_{ext_to}_level")
        else:
            model_name = model_name.replace(f"_{steps}_",f"_{steps}_to_{ext_to}_")
            model_path = model_path.replace(f"_{steps}_",f"_{steps}_to_{ext_to}_")

    if algo_name == algo_name == "ext_dqn" or "norm_guided_dqn" in algo_name:
        algo_name = "dqn"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = env.action_space.n
    action_tensor = torch.tensor(range(num_actions), device=device)
    print(f"Loading {model_name}")

    if exact_model_number is None:
        model = load_model(model_path,algo_name,env=env)
    else:
        model_path = f"{model_path}_{exact_model_number}"
        model = load_model(model_path,algo_name,env=env,exact_match=True)

    stats = evaluate(env,algo_name, model, nr_eps, action_tensor, additional_data_labels,norm_descriptor, norm_helper)
    if extended or norm_guided:
        stats_path = f"pickles/eval_stats/extended/{model_name}"
        if exact_model_number is not None:
            stats_path = f"{stats_path}_{exact_model_number}.pkl"
            save_pickle(stats_path,stats,exact_match=True)
        else:
            save_pickle(stats_path, stats)
    elif norm_helper is not None:
        stats_path = f"pickles/eval_stats/policy_fix/{model_name.replace('dqn',f'dqn_{norm_descriptor}')}"
        if exact_model_number is not None:
            stats_path = f"{stats_path}_{exact_model_number}.pkl"
            save_pickle(stats_path,stats,exact_match=True)
        else:
            save_pickle(stats_path, stats)
    else:
        stats_path = f"pickles/eval_stats/base/{model_name}"
        if exact_model_number is not None:
            stats_path = f"{stats_path}_{exact_model_number}.pkl"
            save_pickle(stats_path,stats,exact_match=True)
        else:
            save_pickle(stats_path, stats)

if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    # mode = int(sys.argv[3]) if "Pacman" not in env_name else sys.argv[3]
    if "Pacman" in env_name:
        mode = sys.argv[3]
    elif "gardener" in env_name:
        split_arg = sys.argv[3].split("-")
        instSize = int(split_arg[0])
        pct_walls = float(split_arg[1])
        pct_plants = float(split_arg[2])
        pct_frogs = float(split_arg[3])
        seed = float(split_arg[4])
        mode = (instSize,pct_walls,pct_plants,pct_frogs,seed)
    elif "sumo" in env_name:
        level = sys.argv[3].split("-")
        mode = get_sumo_level_name(level)
    else:
        raise Exception("Unsupported")
    nr_eps = int(sys.argv[4])
    algo_name = sys.argv[5]
    feature_extractor = sys.argv[6] # "complete"
    ext_to = None
    exact_model_number = None

    norm_descriptor = None
    norm_helper = None
    for arg in sys.argv:
        if "--ext" in arg:
            ext_to = int(arg.replace("--ext",""))
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))
        if "--norm" in arg:
            norm_descriptor, norm_helper = get_norm_helper(env_name,arg, mode)

    additional_data_labels = []
    if "Pacman" in env_name:
        additional_data_labels.append("agent_eaten")
    elif "garden" in env_name:
        additional_data_labels.append("nr-killed-frogs")
        additional_data_labels.append("nr-killed-plants")
    elif "sumo" in env_name:
        additional_data_labels.append("ambulance_waited_too_long")

    complete_policy_eval(algo_name, env_name, mode, steps, nr_eps,
                         feature_extractor,ext_to, exact_model_number,additional_data_labels, norm_descriptor, norm_helper)
