import json
import math
import statistics
import sys
import time
import numpy as np
#import highway_env
import gym_pacman_rules
import torch
from torch.distributions import Categorical

from rule_learning.util import load_pickle, load_model, save_pickle
from env_util import create_environment_and_modelname
from shield.shields import AspShield, RuleChooser, RandomShield
from gym_pacman_rules.envs.featureExtractors import features_dict_to_array
from rule_util import RuleSnapshot
import copy, logging
from collections import deque 


log = logging.getLogger(__name__)




def find_top_k_indices(base_model_name, top_string, return_rew = False):
    top_k = int(top_string.replace("top", ""))
    eval_stats_name = base_model_name.replace("/models/", "/eval_stats/")
    eval_stats = load_pickle(eval_stats_name)
    eval_stats_list = list(eval_stats.items())
    eval_stats_list.sort(key=lambda item: item[0] != -1 and item[1].avg_rew, reverse=True)
    top_k_rule_indices = [i for i, v in eval_stats_list if v.avg_rew > eval_stats[-1].avg_rew][:top_k]
    if return_rew:
        top_k_rule_indices = [i for i, v in eval_stats_list if v.avg_rew > eval_stats[-1].avg_rew][:top_k]
        top_k_rew = [v.avg_rew for i,v in eval_stats_list if v.avg_rew > eval_stats[-1].avg_rew][:top_k]
        return top_k_rule_indices,top_k_rew
    else:
        return top_k_rule_indices


class EvalStats:
    def __init__(self,all_rews,action_changes_list,action_changes_updated_list,wins,all_violations):
        self.all_rews = all_rews
        self.action_changes = action_changes_list
        self.action_changes_updated_list = action_changes_updated_list
        self.cum_rews = []
        self.cum_violations = all_violations
        self.steps =[]
        nr_eps = len(self.all_rews)
        self.action_changes_relation = []
        for i, ac in enumerate(action_changes_list):
            self.action_changes_relation.append(action_changes_updated_list[i]/ac)
        for rewards in all_rews:
            cum_rew = sum(rewards)
            nr_steps = len(rewards)
            self.cum_rews.append(cum_rew)
            self.steps.append(nr_steps)
        self.avg_rew = statistics.mean(self.cum_rews)
        self.avg_violations = statistics.mean(self.cum_violations)
        self.avg_steps = statistics.mean(self.steps)
        self.avg_action_changes = statistics.mean(action_changes_list)
        self.avg_action_changes_relation = statistics.mean(self.action_changes_relation)
        self.stderr_rew = statistics.stdev(self.cum_rews) / math.sqrt(nr_eps)
        self.stderr_violations = statistics.stdev(self.cum_violations) / math.sqrt(nr_eps)
        self.stderr_steps = statistics.stdev(self.steps) / math.sqrt(nr_eps)
        self.stderr_action_changes = statistics.stdev(action_changes_list) / math.sqrt(nr_eps)
        self.stderr_action_changes_relation = statistics.stdev(self.action_changes_relation) / math.sqrt(nr_eps)
        self.wins = wins
        self.nr_wins = sum(wins)

def create_rule_string_for_pos_neg(shield,shield_rule_nrs):
    rule_string = ''
    for shield_rule_nr in shield_rule_nrs:
        if shield_rule_nr < len(shield.pos_rules_list):
            rule_string += shield.pos_rules_list[shield_rule_nr] + "\n"
        else:
            rule_string += shield.neg_rules_list[shield_rule_nr - len(shield.pos_rules_list)] + "\n"
    return rule_string

def change_action(action, pos_triggered,neg_triggered,algo_name,act_logits,action_tensor,triggered_rules, change_type):
    if change_type == "favor_cancel":
        if len(neg_triggered) == 0:
            change_type = "rule_action"
        else:
            change_type = "cancel"
    if change_type == "favor_enforce":
        if pos_triggered is not None:
            change_type = "rule_action"
        else:
            change_type = "cancel"

    (pos_triggered_rules, neg_triggered_rules) = triggered_rules
    pos_triggered_created_rules = list(filter(lambda r : not r[0].mined, pos_triggered_rules))
    neg_triggered_created_rules = list(filter(lambda r : not r[0].mined, neg_triggered_rules))

    if "rule_action" in change_type:
        if action == pos_triggered:
            return None, [] # signal no change
        else:
            corresponding_triggered_rules = list(filter(lambda r : r[0].rule_head.action == pos_triggered, pos_triggered_created_rules))
            return pos_triggered, corresponding_triggered_rules
    elif "cancel" == change_type:
        if len(neg_triggered) == 0:
            return None, []
        if algo_name == "ppo":
            for rule_action in neg_triggered:
                act_logits[rule_action] = -1e6
                return Categorical(logits=act_logits).sample()
        elif "dqn" in algo_name:
            # for some reason, the shape of q values keeps changing, so let's protect against erroneous results and fixed
            # the issue when it occurs again
            if list(act_logits.shape) != [1, action_tensor.shape[0]]:
                act_logits = act_logits.squeeze()
            else:
                assert act_logits.shape == action_tensor.shape
            if len(neg_triggered) < action_tensor.shape[0]:
                for rule_action in neg_triggered:
                    act_logits[rule_action] = -1e6
            corresponding_triggered_rules = list(filter(lambda r : r[0].rule_head.action in neg_triggered, neg_triggered_created_rules))
            return torch.argmax(act_logits).item(), corresponding_triggered_rules
    else:
        raise Exception("Unsupported")


def eval_single_eps(env, algo_name,model,action_tensor, feature_extractor,horizon, shield : AspShield = None,rule_chooser = None,change_type=None,violation_check=None):
    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    win = False
    rewards = []
    use_rule = shield is not None
    action_changes = 0
    action_changes_due_updated_rules = 0
    total_violations = 0
    last_n_states = deque(maxlen=horizon)
    last_n_triggered_rules = deque(maxlen=horizon)
    last_n_triggered_rules.append([])


    rules_snapshot = RuleSnapshot(
        enforceable_rules = shield.enforceable_rules,
        cancelable_rules = shield.cancelable_rules,
    )
    while True:
        if algo_name == "ppo":
            action, _states = model.predict(obs)
            estimated_value, act_logits, _ = policy.evaluate_actions(obs_t, action_tensor)  # .log_prob(action_tensor)
        elif "dqn" in algo_name:
            action, _states = model.predict(obs)
            q_values = policy.q_net(obs_t).squeeze()

            act_logits = q_values  # not logits but treat them as such for now
        else:
            raise Exception("Unsupported")
        if use_rule:
            obs_rules = features_dict_to_array(feature_extractor.getFeatures(env.unwrapped.game.state,action))
            triggers,triggered,triggered_rules= shield.does_rule_trigger(obs_rules,rule_chooser, rules_snapshot)
            if triggered_rules == None:
                triggered_rules = []
            last_n_triggered_rules.append(triggered_rules)
            if triggers:
                (pos_triggered, neg_triggered) = triggered
                changed_action, activated_created_rules = change_action(action,pos_triggered,neg_triggered,algo_name,act_logits, action_tensor,
                                               triggered_rules=triggered_rules, change_type=change_type)
                if changed_action is not None:
                    action_changes += 1
                    action = changed_action
                    if len(activated_created_rules) > 0:
                        action_changes_due_updated_rules += 1


        obs, reward, term, trunc, info = env.step(action)
        last_n_states.append(copy.deepcopy(env.unwrapped.game.state))
        obs_t, vectorized_env = policy.obs_to_tensor(obs)
        obs_t = obs_t.to(action_tensor.device)
        rewards.append(reward)

        if violation_check != None:
            tmp_violations = violation_check(env.unwrapped.game.state)
            if tmp_violations > 0:
                total_violations += tmp_violations
                for j, state in enumerate(last_n_states):
                    if horizon - j - 1 != 0: 
                        log.info(
                            f"{horizon - j - 1} step(s) before violation:\n{state}\n"
                            f"triggered rules:\n\t"
                            f"{'\n\t'.join(f'{r[0]}' for rs in last_n_triggered_rules[j+1] for r in rs)}\n"
                        )
                    else:
                        log.info(
                            f"violation:\n{state}\n"
                        )
    
        if term and reward > 0:
            win = True # TODO check if true for all environments
        if term or trunc:
            break


    return win, rewards, action_changes, action_changes_due_updated_rules, total_violations


def evaluate(env,algo_name, model,nr_eps,action_tensor,feature_extractor, rule_string = '',shield = None,rule_chooser = None, change_type = None, violation_check=None,horizon=1):
    if rule_chooser is not None:
        assert shield is not None
    action_changes_list = []
    action_changes_updated_list = []

    if rule_string != '':
        print(f"Evaluation: {rule_string}" )
    all_rews = []
    wins = []
    all_violations = []
    for i in range(nr_eps):
        win,rewards,action_changes,action_changes_updated,violations = eval_single_eps(env,algo_name,model,action_tensor,feature_extractor,horizon,shield,rule_chooser,change_type,violation_check)
        wins.append(win)
        all_rews.append(rewards)
        all_violations.append(violations)
        action_changes_list.append(action_changes)
        action_changes_updated_list.append(action_changes_updated)

    eval_stats = EvalStats(all_rews,action_changes_list,action_changes_updated_list,wins,all_violations)
    return eval_stats


def setup_shield(env_name,mode,steps_initial,shield_feat,improved,random_shield,
                 exact_model_number = None, algo_name = "dqn", steps_norm=0, updated = False, config_str = ""):
    if improved:
        shield_type = "improved"
    else:
        shield_type = "uncorr"

    if random_shield:
        shield_type = "random"

    if steps_norm == 0:
        shield_name = f"pickles/shields/{shield_type}/" \
                    f"{algo_name}_{env_name.replace('/','_')}_{mode}__{config_str}__feat_{shield_feat}_{steps_initial}_shield"
    else:
        shield_name = f"pickles/shields/{shield_type}/" \
                    f"{algo_name}__{config_str}__{env_name.replace('/','_')}_{mode}_feat_{shield_feat}_{steps_initial}_to_{steps_norm}_shield{'_updated' if updated else ''}"
    
    if exact_model_number is None:
        shield = load_pickle(shield_name)
    else:
        shield_name = f"{shield_name}_{exact_model_number}.pkl"
        shield = load_pickle(shield_name, exact_match=True)
    return shield


def complete_policy_eval(algo_name, env_name, mode, steps, nr_eps, shield_rule,feature_extractor,ext_to,
                         exact_model_number,random_shield):

    env, model_name, model_path = create_environment_and_modelname(algo_name, env_name, mode, steps, feature_extractor=feature_extractor)
    extended = algo_name == "ext_dqn"

    base_model_path = model_path.replace("ext_dqn", "dqn")
    if ext_to is not None:
        if "Berkeley" in env_name:
            model_name = model_name.replace(f"{steps}_level",f"{steps}_to_{ext_to}_level")
            model_path = model_path.replace(f"{steps}_level",f"{steps}_to_{ext_to}_level")
        else:
            model_name = model_name.replace(f"_{steps}",f"_{steps}_to_{ext_to}")
            model_path = model_path.replace(f"_{steps}",f"_{steps}_to_{ext_to}")

    if algo_name == "ext_dqn":
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
    shield = None
    if ext_to is not None:
        # all extended use only the shield with index 1
        shield_exact_model_number = None
    else:
        shield_exact_model_number = exact_model_number

    if random_shield is not None:
        percent = random_shield[0]
        nr_eval = random_shield[1]
        stats_dict = dict()

        base_stats = evaluate(env, algo_name, model, nr_eps, action_tensor,
                         None, None, None, change_type=None)
        stats_dict[-1] = base_stats
        for cancel in [True,False]:
            for i in range(nr_eval):
                shield = RandomShield(num_actions,percent_random =percent, cancel = cancel, cancel_stop = 0.66)
                cancel_str = "canceling" if cancel else "enforcing"
                stats = evaluate(env,algo_name, model, nr_eps, action_tensor,
                                 f"random run {i} with {cancel_str}:", shield, None,change_type="favor_cancel")
                stats_store_obj = (shield.triggers,stats)
                stats_dict[(i,cancel)] = stats_store_obj
        stats_path = f"pickles/eval_stats/random/{percent}/{model_name}"

        if exact_model_number is None:
            save_pickle(stats_path,stats_dict)
        else:
            save_pickle(f"{stats_path}_{exact_model_number}_{int(time.time())}.pkl",stats_dict,exact_match=True)
    elif shield_rule is not None:
        shield_feat = shield_rule[0]
        shield = setup_shield(env_name,mode,steps,shield_feat,shield_rule[1]=="from_shield",
                              shield_rule[2] == "random",
                              exact_model_number=shield_exact_model_number)

        if shield_rule[1] == "from_shield":
            assert hasattr(shield, 'enforceable_rules')
            assert hasattr(shield, 'cancelable_rules')
            if shield_rule[2] == "none" or shield_rule[2] == "random":
                eval_stats_dict = dict()

                stats = evaluate(env, algo_name, model, nr_eps, action_tensor, None, None)
                eval_stats_dict[-1] = stats # -1 is standard stats without enforced rules
                rule_chooser = RuleChooser(shield)
                for index,pos_rule in enumerate(sorted(shield.enforceable_rules.keys())):
                    rule_string = f"Enforcing ({index}) from {pos_rule} adapted  \n"
                    adapted_rules = shield.enforceable_rules[pos_rule]
                    rule_string += "\n".join(adapted_rules)
                    rule_chooser.set_rules_list([index])
                    stats = evaluate(env,algo_name,model,nr_eps, action_tensor,rule_string,shield,rule_chooser,"rule_action")
                    eval_stats_dict[index] = stats
                for index,neg_rule in enumerate(sorted(shield.cancelable_rules.keys())):
                    overall_index = index + len(shield.enforceable_rules.keys())
                    rule_string = f"Cancelling ({index}) from {neg_rule} adapted  \n"
                    adapted_rules = shield.cancelable_rules[neg_rule]
                    rule_string += "\n".join(adapted_rules)
                    rule_chooser.set_rules_list([len(shield.enforceable_rules) + index])
                    stats = evaluate(env,algo_name,model,nr_eps, action_tensor,rule_string,shield,rule_chooser,"cancel")
                    eval_stats_dict[overall_index] = stats
                if shield_rule[2] == "random":
                    stats_path = f"pickles/eval_stats/random_rules/{model_name}"
                else:
                    stats_path = f"pickles/eval_stats/{model_name}"


                if exact_model_number is None:
                    save_pickle(stats_path, eval_stats_dict)
                else:
                    stats_path = f"{stats_path}_{exact_model_number}.pkl"
                    save_pickle(stats_path, eval_stats_dict, exact_match=True)

            elif shield_rule[2].startswith("top") or shield_rule[2].startswith("[")\
                    or shield_rule[2].startswith("find_comb") or shield_rule[2].startswith("comb"):

                rule_chooser = RuleChooser(shield)

                if shield_rule[2].startswith("top") or shield_rule[2].startswith("[") or shield_rule[2].startswith("comb"):
                    if shield_rule[2].startswith("top"):
                        top_k_rule_indices = find_top_k_indices(base_model_path, shield_rule[2])
                        rules_list = top_k_rule_indices
                        print(f"{len(top_k_rule_indices)} rules selected: {rules_list}")
                    elif shield_rule[2].startswith("comb"):
                        comb_stats_path = f"pickles/eval_stats/combination/{model_name}"
                        comb, comb_stats = load_pickle(comb_stats_path)
                        rules_list = comb
                        print(f"Evaluation with combination list {comb}.")
                    else:
                        rules_list = json.loads(shield_rule[2])

                    adapted_rules,rule_keys = rule_chooser.select_rules(rules_list,return_keys=True)
                    rule_string = f"Enforcing ({rules_list}) from {' '.join(rule_keys)} adapted \n"
                    rule_string += "\n".join(adapted_rules)
#
                    rule_chooser.set_rules_list(rules_list)
                    stats = evaluate(env,algo_name,model,nr_eps*2,action_tensor,rule_string,
                                     shield,rule_chooser,"favor_cancel")

                    stats_path = f"pickles/eval_stats/selected_{shield_rule[2]}/{model_name}"
                    save_pickle(stats_path, stats)
                else:
                    rule_nr_limit = 20
                    if rule_nr_limit >= len(rule_chooser.sorted_enforce_rules) + len(rule_chooser.sorted_cancel_rules):
                        # starts from cancelable rules
                        selection_schedule= list(range(len(rule_chooser.sorted_enforce_rules),len(rule_chooser.sorted_enforce_rules)
                                                       + len(rule_chooser.sorted_cancel_rules)))
                        selection_schedule.extend(list(range(len(rule_chooser.sorted_enforce_rules))))
                    else:
                        if exact_model_number is not None:
                            individual_eval_stats = load_pickle(f"pickles/eval_stats/{model_name}_{exact_model_number}.pkl",exact_match=True)
                        else:
                            individual_eval_stats = load_pickle(f"pickles/eval_stats/{model_name}")

                        selection_schedule = []
                        base_rew = individual_eval_stats[-1].avg_rew
                        base_err = individual_eval_stats[-1].stderr_rew
                        for rule_index in individual_eval_stats.keys():
                            if rule_index != -1:
                                if individual_eval_stats[rule_index].avg_rew + individual_eval_stats[rule_index].stderr_rew >= base_rew - base_err:
                                    selection_schedule.append((rule_index, individual_eval_stats[rule_index].avg_rew))
                        selection_schedule.sort(key=lambda x : x[1],reverse=True)
                        selection_schedule = [ri for ri,rew in selection_schedule][:rule_nr_limit]

                    stats = evaluate(env, algo_name, model, nr_eps, action_tensor)

                    # greedy selection of rules
                    rules_list = []
                    avg_rew = stats.avg_rew
                    print(f"Starting from average reward {avg_rew}")
                    print(f"Selection schedule: {selection_schedule}")
                    best_stats = stats
                    for ri in selection_schedule:
                        new_rules_list = rules_list + [ri]
                        adapted_rules,rule_keys = rule_chooser.select_rules(new_rules_list,return_keys=True)
                        rule_string = f"Enforcing ({new_rules_list}) from {' '.join(rule_keys)} adapted \n"
                        rule_string += "\n".join(adapted_rules)

                        rule_chooser.set_rules_list(new_rules_list)
                        stats = evaluate(env,algo_name,model,nr_eps, action_tensor,rule_string,
                                         shield,rule_chooser,"favor_cancel")
                        new_avg_rew = stats.avg_rew
                        if new_avg_rew > avg_rew:
                            print(f"Improved reward from {avg_rew} to {new_avg_rew} with combination {rules_list}")
                            rules_list = new_rules_list
                            best_stats = stats
                            avg_rew = new_avg_rew
                        else:
                            print(f"Decreased reward from {avg_rew} to {new_avg_rew} with {ri} added in combination {new_rules_list}")
                    print(f"Final average reward {avg_rew} with combination {rules_list}")
                    stats_path = f"pickles/eval_stats/combination/{model_name}"
                    if exact_model_number is None:
                        save_pickle(stats_path, (rules_list,best_stats))
                    else:
                        stats_path = f"{stats_path}_{exact_model_number}.pkl"
                        save_pickle(stats_path, (rules_list,best_stats), exact_match=True)

            else:
                raise Exception("No other rule selection mode implemented.")


        else:
            raise Exception("The only mode currently supported is 'from_shield' for rule-guided execution.")
    else:
        stats = evaluate(env,algo_name, model, nr_eps, action_tensor, None, None, None)
        if extended:
            stats_path = f"pickles/eval_stats/extended/{model_name}"
            save_pickle(stats_path, stats)

if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    mode = int(sys.argv[3]) if "Pacman" not in env_name else sys.argv[3]
    nr_eps = int(sys.argv[4])
    shield_rule = None
    algo_name = sys.argv[5]
    ext_to = None
    exact_model_number = None

    random_shield = None
    for arg in sys.argv:
        if "--shield" in arg:
            shield_rule_tuple_str = arg.replace("--shield","")
            split_str = shield_rule_tuple_str.split("-")
            shield_feat = int(split_str[0])
            rule_nr = split_str[1]
            change_type = split_str[2]
            shield_rule = (shield_feat,rule_nr,change_type)
            print(shield_rule)
        if "--ext" in arg:
            ext_to = int(arg.replace("--ext",""))
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))
        if "--rand" in arg:
            random_shield_str = arg.replace("--rand","")
            split_str = random_shield_str.split("-")
            percent = int(split_str[0])
            nr_eval = int(split_str[1])
            random_shield = (percent,nr_eval)

    feature_extractor = "extended-8"
    complete_policy_eval(algo_name, env_name, mode, steps, nr_eps, shield_rule,
                         feature_extractor,ext_to, exact_model_number,random_shield)
