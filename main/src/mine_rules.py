import hydra
from omegaconf import DictConfig
from legible import feature_and_rule_learn

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def mine_rules(cfg : DictConfig) -> None:
    feature_and_rule_learn.select_features_and_learn_rules(cfg.env.name, cfg.training.steps_initial, cfg.env.level, cfg.rules.episodes,
                                                           cfg.rules.nr_features, cfg.rules.lime_test_size, cfg.rules.compute_correlation,
                                                            "norm_guided_dqn", cfg.env.feature_extractor, exact_model_number="2", steps_norm=cfg.training.steps_norm,
                                                            norm_descriptor="vegetarian_2_5")
    # TODO change descriptor
    

if __name__ == "__main__":
    mine_rules()