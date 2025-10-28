import hydra
from omegaconf import DictConfig, OmegaConf
from oftendeeprl import train_pacman

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train_base_model(cfg : DictConfig) -> None:
    train_pacman.train(cfg.env.name, cfg.training.algorithm, cfg.env.feature_extractor, cfg.training.steps, cfg.env.level)

    

if __name__ == "__main__":
    train_base_model()