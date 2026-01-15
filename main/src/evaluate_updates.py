import hydra
from omegaconf import DictConfig
from util import get_shield_number, get_model_number, get_feature_extractor
from rule_util import set_rules
from legible.env_util import create_environment_and_modelname_for_oftendeeprl
from legible.evaluate_policy import setup_shield, evaluate
from legible.rule_learning.util import load_model, save_pickle
from legible.shield.shields import RuleChooser 


import sys, logging, torch
import check_norms

log = logging.getLogger(__name__)

def setup(cfg):
    norm_descriptor = f"{'_'.join(cfg.norms)}"
    config_str = f"{norm_descriptor}__{str(cfg.asp.horizon)}_{str(cfg.asp.radius)}"

    env, model_name, model_path = create_environment_and_modelname_for_oftendeeprl("norm_guided_dqn", cfg.env.name,
                                                                                    cfg.env.level, config_str, 
                                                                                    cfg.training.steps_initial, cfg.training.steps_norm,
                                                                                    cfg.training.feature_extractor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_actions = env.action_space.n
    action_tensor = torch.tensor(range(num_actions), device=device)

    model_number = get_model_number(cfg)
    model = load_model(model_path + f"_{model_number}","norm_guided_dqn",env=env,exact_match=True)
    shield_number = get_shield_number(cfg)

    obs, info = env.reset()
    feature_extractor = get_feature_extractor(cfg.rules.feature_extractor,env.unwrapped.layout.height, env.unwrapped.layout.width)
    features = feature_extractor.getFeatures(env.unwrapped.game.state,None)
    ordered_features = sorted(list(features.items()),key=lambda x: x[0])
    feature_to_index = {
        name: idx for idx, name in enumerate(features)
    }
    ordered_feature_indices = [feature_to_index[fv[0]] for fv in ordered_features]


    #TODO move to util?
    shield_initial = setup_shield(cfg.env.name, cfg.env.level, cfg.training.steps_initial, cfg.rules.nr_features, False, False, exact_model_number=shield_number,
                          steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn",horizon=cfg.asp.horizon,radius=cfg.asp.radius)
    
    shield_updated = setup_shield(cfg.env.name, cfg.env.level, cfg.training.steps_initial, cfg.rules.nr_features, False, False, exact_model_number=shield_number,
                          steps_norm=cfg.training.steps_norm, algo_name="norm_guided_dqn", updated=True,horizon=cfg.asp.horizon,radius=cfg.asp.radius)

    if shield_initial is None:
        sys.exit("Could not load initial shield, check if it exists.")
    if shield_updated is None:
        sys.exit("Could not load updated shield, check if it exists.")

    shield_initial.feature_indices = ordered_feature_indices
    shield_updated.feature_indices = ordered_feature_indices
    shield_initial, rule_chooser_initial = set_rules(shield_initial)

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
        return check_norms.num_violations_detected(cfg.norms, state)
    
    #logging.disable(logging.CRITICAL)
    stats = evaluate(env, cfg.training.algorithm, model, cfg.eval.nr_episodes, action_tensor, feature_extractor, '',shield, rule_chooser, "favor_enforce",count_violations)
    #logging.disable(logging.NOTSET)

    log.info(f"Nr. wins: {stats.nr_wins}")
    log.info(f"Avg. reward: {stats.avg_rew} with SE {stats.stderr_rew}")
    log.info(f"Avg. violations: {stats.avg_violations} with SE {stats.stderr_violations}")
    log.info(f"Avg. steps: {stats.avg_steps} with SE {stats.stderr_steps}")
    log.info(f"Avg. action changes: {stats.avg_action_changes} with SE {stats.stderr_action_changes}")
    log.info(f"Avg. relation updated / mined rules: {stats.avg_action_changes_relation} with SE {stats.stderr_action_changes_relation}")
    stats_path = f"pickles/eval_stats/{model_name}_{get_model_number(cfg)}_{'_updated' if updated else ''}"
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
