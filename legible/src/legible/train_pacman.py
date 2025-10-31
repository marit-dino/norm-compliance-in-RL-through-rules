import gym_pacman_rules
import gymnasium as gym
import torch
from gymnasium.wrappers import TransformReward
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

from rule_learning.util import save_model



def train(env_name,algo,feature_extractor, n_steps,level, tb_name = None):
    env = make_vec_env(lambda : create_pacman_env(env_name, feature_extractor = feature_extractor, level=level, render_mode="none"
                                                  ,scale=algo=="ppo"), 8)

    if algo == "ppo":
        policy_kwargs_ppo = dict(
            net_arch = dict(pi=[256]*2, vf=[256]*2),
            activation_fn = torch.nn.ReLU,share_features_extractor =False
        )
    else:
        policy_kwargs_dqn = dict(
            net_arch=[256]*2,
            activation_fn=torch.nn.ReLU,
        )

    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None

    if algo == "ppo":
        model = PPO("MlpPolicy", env, verbose =1, tensorboard_log=tb_path, policy_kwargs=policy_kwargs_ppo,
                    vf_coef= 0.75, ent_coef= 0.00,clip_range=0.5,
                    learning_rate=2.5e-4,
                    batch_size=256,gamma=0.9,
                    n_epochs=10,n_steps=256)
    else:
        model = DQN("MlpPolicy", env, verbose =1, tensorboard_log=tb_path, policy_kwargs=policy_kwargs_dqn,
                    batch_size=256, buffer_size=50_000, exploration_fraction=0.5, gamma = 0.95,gradient_steps=-1
                   )
    model.learn(n_steps)

    return model


def create_pacman_env(env_name,feature_extractor, level, render_mode,scale = False):
    env_name = env_name + "-legible"
    env = gym.make(env_name, layout=level,features=feature_extractor,render_mode=render_mode)
    if scale:
        env = TransformReward(env, lambda r: 0.01*r)
    return env

def run_trained_model(env_name,model,feature_extractor, level):
    env = create_pacman_env(env_name,feature_extractor=feature_extractor, level=level, render_mode="human")

    obs,info = env.reset()
    while True:
        action, _states = model.predict(obs)
        action = action.item()
        obs, rewards, term,trunc, info = env.step(action)
        env.render()
        if term or trunc:
            obs, info = env.reset()

def train_pacman(algo,env_name, steps, level,feature_extractor, run_afterwards=False):

    model_name = f"pickles/models/{algo}_{env_name.replace('/','_')}_{steps}_level_{level}_{feature_extractor}"
    tb_name = f"{algo}_{env_name}_mode_{level}_{feature_extractor}"

    model = train(env_name,algo,feature_extractor = feature_extractor, n_steps=steps,
                      tb_name = tb_name, level = level)
    save_model(model_name, model)
    if run_afterwards:
        input("press")
        run_trained_model(env_name,model,feature_extractor = feature_extractor, level=level)

if __name__ == "__main__":
    import sys
    algo = "dqn"
    env_name = sys.argv[1]
    steps = int(sys.argv[2]) #10000000
    level = sys.argv[3]
    feature_extractor = "extended-8"
    train_pacman(algo,env_name, steps, level,feature_extractor)