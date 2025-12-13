import hydra
from omegaconf import DictConfig
from legible import feature_and_rule_learn
from os import listdir
from os.path import isfile, join
import sys

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def mine_rules(cfg : DictConfig) -> None:

    norm_descriptor = f"{'_'.join(cfg.norms)}"
    config_str = f"{norm_descriptor}__{str(cfg.asp.horizon)}_{str(cfg.asp.radius)}"


    model_number = ""
    if cfg.rules.model_number is None:
        model_name = f"norm_guided_dqn__{config_str}__{cfg.env.name.replace('/', '_')}_{cfg.training.steps_initial}_to_{cfg.training.steps_norm}_level_{cfg.env.level}_{cfg.training.feature_extractor}"
        norm_guided_models = [f for f in listdir('./pickles/models') if isfile(join('./pickles/models', f)) 
                            and f.startswith(model_name)]
        if len(norm_guided_models) == 0:
            sys.exit("No policy found that matches the provided parameters.")
        model_number = sorted(norm_guided_models)[-1].removesuffix(".zip").rsplit("_", 1)[-1]
    else: 
        model_number = cfg.rules.model_number

    feature_and_rule_learn.select_features_and_learn_rules(cfg.env.name, cfg.training.steps_initial, cfg.env.level, cfg.rules.episodes,
                                                           cfg.rules.nr_features, cfg.rules.lime_test_size, cfg.rules.compute_correlation,
                                                            "norm_guided_dqn", cfg.training.feature_extractor, exact_model_number=model_number, steps_norm=cfg.training.steps_norm,
                                                            norm_descriptor=config_str, feature_extractor_rules=cfg.rules.feature_extractor)    

if __name__ == "__main__":
    mine_rules()