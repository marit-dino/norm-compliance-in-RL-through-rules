import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from torch import nn


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        print(observation_space)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        print(self.cnn)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class MinigridDeflattenFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, img_obs_space : gym.Space,
                 view_size = 7, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = img_obs_space.shape[2] # we need to transpose the image part
        # which is actually done through permut in pytorch, rather than transpose as in numpy
        self.view_size = view_size
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            # permute include first dimensions which is batch size
            n_flatten = self.cnn(torch.as_tensor(img_obs_space.sample()[None]).float().permute((0,3, 1, 2))).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten+1, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        img_part = observations[:,:-1]
        dir_part = observations[:,-1]
        img_part_deflattened = img_part.reshape((batch_size,self.view_size,self.view_size,3)).permute((0,3, 1, 2))
        cnn_out = self.cnn(img_part_deflattened)
        dir_part = dir_part.unsqueeze(1)
        combined = torch.cat((cnn_out,dir_part),dim=1)
        return self.linear(combined)
