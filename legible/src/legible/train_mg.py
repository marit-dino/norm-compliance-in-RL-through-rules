import sys

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from mg.env_utils import create_mg_env
from mg.feature_extractors import MinigridDeflattenFeaturesExtractor
from rule_learning.util import save_model, save_pickle, save_stored_training_data


def train_ppo(env_name,n_steps, tb_name = None):
    buffer_list = []
    dummy_env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    img_obs_space = ImgObsWrapper(dummy_env).observation_space
    env = make_vec_env(lambda : create_mg_env(env_name, buffer_list=buffer_list,store_data = False), 8)

    policy_kwargs = dict(
        features_extractor_class=MinigridDeflattenFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128, img_obs_space=img_obs_space),
    )

    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None
    model = PPO("CnnPolicy", env, verbose =1, policy_kwargs=policy_kwargs, tensorboard_log=tb_path)
    model.learn(n_steps)

    save_stored_training_data(buffer_list, env_name, n_steps)
    return model

def run_trained_model(env_name,model):
    env = create_mg_env(env_name, render_mode="human")

    obs,info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term,trunc, info = env.step(action)
        env.render()
        if term or trunc:
            obs, info = env.reset()

def main(env_name,steps,run_afterwards = False):
    # env_name = "MiniGrid-walls-and-lava-2-2"
    #  = 1000000
    model_name = f"pickles/models/ppo_{env_name}_{steps}"
    tb_name = f"ppo_{env_name}_unshielded"

    model = train_ppo(env_name,n_steps=steps, tb_name = tb_name)
    save_model(model_name, model)
    if run_afterwards:
        run_trained_model(env_name,model)

if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    main(env_name,steps)