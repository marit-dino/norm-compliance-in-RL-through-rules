import hydra
from omegaconf import DictConfig
from util import get_shield_number, get_model_number, get_feature_extractor, setup_model, order_feature_indices
from rule_util import set_rules
from legible.evaluate_policy import setup_shield, evaluate
from legible.rule_learning.util import save_pickle
from legible.shield.shields import RuleChooser 


import sys, logging, torch
import check_norms

log = logging.getLogger(__name__)

def setup(cfg):
    env, model, model_name, action_tensor = setup_model(cfg)

    shield_number = get_shield_number(cfg)
    config_str = f"{cfg.norm.id}__{str(cfg.asp.horizon)}_{str(cfg.asp.radius)}"

    shield_initial = setup_shield(cfg.env.name, cfg.env.level, cfg.training.steps_initial, cfg.rules.nr_features, False, False, exact_model_number=shield_number,
                          steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn", config_str=config_str)
    
    shield_updated = setup_shield(cfg.env.name, cfg.env.level, cfg.training.steps_initial, cfg.rules.nr_features, False, False, exact_model_number=shield_number,
                          steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn", updated=True, config_str=config_str)

    if shield_initial is None:
        sys.exit("Could not load initial shield, check if it exists.")
    if shield_updated is None:
        sys.exit("Could not load updated shield, check if it exists.")

    order_feature_indices(shield_initial, env, cfg.rules.feature_extractor)
    order_feature_indices(shield_updated, env, cfg.rules.feature_extractor)
    rule_chooser_initial = set_rules(shield_initial)

    rule_chooser_updated = RuleChooser(shield_updated)
    rule_chooser_updated.set_rules_list(list(range(0,len(list(shield_updated.enforceable_rules.keys()) + list(shield_updated.cancelable_rules.keys()))))) 
    
    assert hasattr(shield_initial, 'enforceable_rules')
    assert hasattr(shield_initial, 'cancelable_rules')
    assert hasattr(shield_updated, 'enforceable_rules')
    assert hasattr(shield_updated, 'cancelable_rules')

    return env, model, model_name, action_tensor, shield_initial, rule_chooser_initial, shield_updated, rule_chooser_updated

def evaluate_rules(env, model, model_name, rule_chooser, shield, action_tensor, cfg, updated=False):
    feature_extractor = get_feature_extractor(cfg.rules.feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)
    
    def count_violations(state):
        return check_norms.num_violations_detected(cfg.norm.id, state)
    
    stats = evaluate(env, cfg.training.algorithm, model, cfg.eval.nr_episodes, action_tensor, feature_extractor, '',shield, rule_chooser, "favor_enforce",count_violations,horizon=cfg.asp.horizon,norm=cfg.norm.id)

    log.info(f"Nr. wins: {stats.nr_wins}")
    log.info(f"Avg. reward: {stats.avg_rew} with SE {stats.stderr_rew}")
    log.info(f"Avg. violations: {stats.avg_violations} with SE {stats.stderr_violations}")
    log.info(f"Avg. steps: {stats.avg_steps} with SE {stats.stderr_steps}")
    log.info(f"Avg. action changes: {stats.avg_action_changes} with SE {stats.stderr_action_changes}")
    if updated:
        log.info(f"Avg. relation updated / mined rules: {stats.avg_action_changes_relation} with SE {stats.stderr_action_changes_relation}")
    if cfg.norm.id == "ctd":
        log.info(f"Avg. CTD violations: {stats.avg_ctd_violations} with SE {stats.stderr_ctd_violations}")
    if cfg.norm.id == "permissive":
        log.info(f"Avg. permitted eaten ghosts: {stats.avg_permitted_eaten_ghosts} with SE {stats.stderr_permitted_eaten_ghosts}")
    stats_path = f"pickles/eval_stats/{model_name}_{get_model_number(cfg)}_{'_updated' if updated else ''}.pkl"
    save_pickle(stats_path, stats, exact_match=True)




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    env, model, model_name, action_tensor, shield_initial, rule_chooser_initial, shield_updated, rule_chooser_updated = setup(cfg)
    log.info("Original rules:")
    evaluate_rules(env, model, model_name, rule_chooser_initial, shield_initial, action_tensor, cfg)
    log.info("Updated rules:")
    evaluate_rules(env, model, model_name, rule_chooser_updated, shield_updated, action_tensor, cfg, True)
   

if __name__ == "__main__":
    main()
