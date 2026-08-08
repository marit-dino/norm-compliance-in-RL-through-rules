import time

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from evaluate_policy import setup_shield
from rule_learning.util import load_pickle
from shield.shields import RuleChooser
from evaluate_policy import EvalStats
from train_pacman import create_pacman_env

env_name = "BerkeleyPacman-v0"
level = "smallClassic_no_capsules"
steps = 2500_000
shield_feat = 69
feature_extractor = "extended-8"
stats_path = f"pickles/eval_stats/dqn_{env_name}_{steps}_level_{level}_{feature_extractor}"
env = create_pacman_env(env_name, feature_extractor = feature_extractor, level=level, render_mode="human")
dqn_env = lambda : create_pacman_env(env_name, feature_extractor = feature_extractor, level=level, render_mode="none")
policy_kwargs_dqn = dict(
    net_arch=[256] * 2,
    activation_fn=torch.nn.ReLU,
)
model = DQN("MlpPolicy", make_vec_env(dqn_env,1), verbose=1, tensorboard_log=None, policy_kwargs=policy_kwargs_dqn,
            batch_size=256, buffer_size=50_000, exploration_fraction=0.5, gamma=0.95, gradient_steps=-1
            )

base_model_name = f"pickles/models/dqn_{env_name.replace('/', '_')}_{steps}_level_{level}_{feature_extractor}_1"
model.set_parameters(base_model_name)


n_eps = 2
for e in range(n_eps):
    obs, info = env.reset()
    while True:
        input("press")
        action, _states = model.predict(obs)
        action = action.item()
        obs, rewards, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            break