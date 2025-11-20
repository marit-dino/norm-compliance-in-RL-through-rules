import hydra
from omegaconf import DictConfig
from legible.env_util import create_environment_and_modelname_for_oftendeeprl
from legible.evaluate_policy import setup_shield, eval_single_eps, EvalStats
from legible.shield.shields import RuleChooser 
from legible.create_rules_pacman import string_to_rule
from legible.shield.create_rules_common import turn_rules_to_str
from legible.rule_learning.util import load_model
import torch
import sys
from os import listdir
from os.path import isfile, join
import os

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def update_rules(cfg : DictConfig) -> None:
    norm_descriptor = cfg.norm + "_" + str(cfg.asp.horizon) + "_" + str(cfg.asp.radius)

    env, model_name, model_path = create_environment_and_modelname_for_oftendeeprl("norm_guided_dqn", cfg.env.name,
                                                                                    cfg.env.level, norm_descriptor, 
                                                                                    cfg.training.steps_initial, cfg.training.steps_norm,
                                                                                    cfg.training.feature_extractor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = env.action_space.n
    action_tensor = torch.tensor(range(num_actions), device=device)

    model_number = get_model_number(cfg, norm_descriptor)
    model = load_model(model_path + f"_{model_number}","norm_guided_dqn",env=env,exact_match=True)
    shield_number = get_shield_number(cfg)

    #TODO what is difference between uncorr and improved?
    shield = setup_shield(cfg.env.name, cfg.env.level, cfg.training.steps_initial, cfg.rules.nr_features, False, False, exact_model_number=shield_number,
                           steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn")

    if shield is None:
        sys.exit("Could not load shield, check if it exists.")

    shield, rule_chooser = set_rules(shield)
    assert hasattr(shield, 'enforceable_rules')
    assert hasattr(shield, 'cancelable_rules')

    #Todo set up rule chooser

    # TODO until convergence

    all_rews = []
    wins = []
    action_changes_list = []

    for i in range(50):
        win, rewards,action_changes = eval_single_eps(env, "norm_guided_dqn", model, action_tensor, shield, rule_chooser, change_type="favor_cancel")
        wins.append(win)
        all_rews.append(rewards)
        action_changes_list.append(action_changes)

    eval_stats = EvalStats(all_rews,action_changes_list,wins)
    print(f"Nr. wins: {eval_stats.nr_wins}")
    print(f"Avg. reward: {eval_stats.avg_rew} with SE {eval_stats.stderr_rew}")
    print(f"Avg. steps: {eval_stats.avg_steps} with SE {eval_stats.stderr_steps}")
    print(f"Avg. action changes: {eval_stats.avg_action_changes} with SE {eval_stats.stderr_action_changes}")


# adapted from legible
def set_rules(shield):
    enforceable_rules = dict()
    cancelable_rules = dict()
    for pos_rule in shield.pos_rules_list:
        if len(pos_rule.strip()) == 0:
            continue
        enforceable_rules[pos_rule] = [string_to_rule(pos_rule)]
    for neg_rule in shield.neg_rules_list:
        if len(neg_rule.strip()) == 0:
            continue
        cancelable_rules[neg_rule] = [string_to_rule(neg_rule)]
    shield.enforceable_rules = turn_rules_to_str(enforceable_rules) if len(enforceable_rules) > 0 else dict()
    shield.cancelable_rules = turn_rules_to_str(cancelable_rules) if len(cancelable_rules) > 0 else dict()
    rule_chooser = RuleChooser(shield)
    rule_chooser.set_rules_list(list(range(0, len(list(enforceable_rules.keys()) + list(cancelable_rules.keys())))))
    return shield, rule_chooser
    

def get_model_number(cfg, norm_descriptor):
    if cfg.rules.model_number is None:
        model_name = f"norm_guided_dqn__{norm_descriptor}__{cfg.env.name.replace('/', '_')}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_level_{cfg.env.level}_{cfg.training.feature_extractor}"
        norm_guided_models = [f for f in listdir('../pickles/models') if isfile(join('../pickles/models', f)) 
                            and f.startswith(model_name)]
        if len(norm_guided_models) == 0:
            sys.exit("No policy found that matches the provided parameters.")
        return sorted(norm_guided_models)[-1].removesuffix(".zip").rsplit("_", 1)[-1]
    else: 
       return cfg.rules.model_number
    
def get_shield_number(cfg):
    if cfg.rules.shield_number is None:
        shield_name = f"norm_guided_dqn_{cfg.env.name.replace('/', '_')}_{cfg.env.level}_feat_{cfg.rules.nr_features}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_shield"
        shields = [f for f in listdir('pickles/shields/uncorr') if f.startswith(shield_name)]
        if len(shields) == 0:
            print(shield_name)
            sys.exit("No shield found that matches the provided parameters.")
        return sorted(shields)[-1].removesuffix(".pkl").rsplit("_", 1)[-1]
    else: 
       return cfg.rules.shield_number

if __name__ == "__main__":
    update_rules()