import hydra
from omegaconf import DictConfig, OmegaConf
from oftendeeprl.train_pacman import train_pacman
from oftendeeprl.extended_training import parse_level, setup_and_ext_train, get_norm_helper

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_base_model(cfg : DictConfig) -> None:
    # train initial policy
    policy_name = train_pacman(cfg.training.algorithm, cfg.env.name, cfg.training.steps_initial, cfg.env.level, cfg.training.feature_extractor)
    level = parse_level(cfg.env.name, cfg.env.level)

    argument_str = "--norm" + str(cfg.asp.horizon) + "-" + str(cfg.asp.radius) + "-" + str(cfg.norm == "vegetarian")
    norm_descriptor, norm_helper = get_norm_helper(cfg.env.name, argument_str, cfg.env.level)
    # train on norms
    model_number = policy_name.removesuffix(".zip").rsplit("_", 1)[-1]
    setup_and_ext_train(cfg.env.name, cfg.training.steps_initial, cfg.training.steps_norm, level, cfg.training.feature_extractor,
                        norm_helper, norm_descriptor = norm_descriptor, exact_model_number=model_number, 
                        ng_margin=cfg.training.ng_margin,norm_violation_filtering=cfg.training.norm_violation_filtering)
    

if __name__ == "__main__":
    train_base_model()