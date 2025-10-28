import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from torch import nn

class MinigridFeaturesExtractorNonDict(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space,  view_size,features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        print(observation_space)
        n_input_channels = 1 #observation_space.spaces["image"].shape[0]
        self.view_size = view_size

        self.device = "cuda"
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (3, 3)),  # Input: (1, 7, 7)
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),  # Input: (16, 6, 6)
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),  # Input: (32, 5, 5)
            nn.ReLU(),
            nn.Flatten(),  # Output: (64 * 4 * 4) = 1024
        ).to(self.device)
        print(self.cnn)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            # n_flatten = self.cnn(torch.as_tensor(observation_space.spaces["image"].sample()[None]).float()).shape[1]
            sample = torch.as_tensor(observation_space.sample()[None]).to(self.device)
            unflattened = unflatten_obs_square(sample,self.view_size,device=self.device)
            print(unflattened.device)
            n_flatten = self.cnn(unflattened).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),  # input: n_flatten (1024), output:features_dim(128)
            nn.ReLU()
        )

        combined_input_dim = features_dim + 1
        self.fc = nn.Sequential(
            nn.Linear(combined_input_dim, features_dim * 2),  # input: 128+1, output 128*2
            nn.Linear(features_dim * 2, features_dim),  # input: 128*2, output: 128
            nn.ReLU()
        )
        print(self.fc)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_output = self.linear(self.cnn(unflatten_obs_square(observations,self.view_size,device=self.device)))
        combined_input = torch.cat((cnn_output, observations[:,-1].unsqueeze(1)), dim=1)

        return self.fc(combined_input)

def unflatten_obs_square(observations,view_size,device):
    obs_array = torch.zeros((observations.shape[0], 1, view_size, view_size),device=device)
    for i in range(view_size):
        for j in range(view_size):
            obs_array[:,0,i,j] = observations[:,i*view_size + j]
    return obs_array


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        # cnn (out_dim 1064) --> linea (out dim = features_dim = 128)\
        #                                                             -> fully connected 128 - 128*2 -> fully connected 128*2 - 128
        # goal_direction --------------------------------------------/

        super().__init__(observation_space, features_dim)
        print(observation_space)
        n_input_channels = observation_space.spaces["image"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (3, 3)), # Input: (1, 7, 7)
            nn.ReLU(),
            nn.Conv2d(16, 32, (3, 3)), # Input: (16, 6, 6)
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)), # Input: (32, 5, 5)
            nn.ReLU(),
            nn.Flatten(), # Output: (64 * 4 * 4) = 1024
        )
        print(self.cnn)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            # n_flatten = self.cnn(torch.as_tensor(observation_space.spaces["image"].sample()[None]).float()).shape[1]
            n_flatten = self.cnn(torch.as_tensor(observation_space.spaces["image"].sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
                    nn.Linear(n_flatten, features_dim), # input: n_flatten (1024), output:features_dim(128)
                    nn.ReLU()
                    )
        
        combined_input_dim = features_dim + 1
        self.fc = nn.Sequential(
                    nn.Linear(combined_input_dim, features_dim*2), # input: 128+1, output 128*2
                    nn.Linear(features_dim*2, features_dim), # input: 128*2, output: 128
                    nn.ReLU()
                    )
        print(self.fc)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        cnn_output = self.linear(self.cnn(observations["image"]))

        combined_input = torch.cat((cnn_output, observations["goal_direction"]), dim=1)

        return self.fc(combined_input)
