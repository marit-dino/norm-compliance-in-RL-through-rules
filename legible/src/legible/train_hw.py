import sys

import highway_env
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from hw.hw_util import create_hw_env, env_kwargs
from rule_learning.util import save_model, load_model


def create_env(**kwargs):
    return create_hw_env(buffer_list=[], store_data=True, **kwargs)


def main(env_name,n_steps,run_afterwards):
    # env_name = "highway-fast-v0"
    # Parallel environments

    model_name = f"pickles/models/dqn_{env_name.replace('/','_')}_{n_steps}"
    n_cpu = 8
    env_kwargs["id"] = env_name


    env = make_vec_env(
        create_env,
        n_envs=n_cpu,
        seed=0,
        env_kwargs=env_kwargs,
    )
    tb_log = "tb_log/" + model_name.replace("pickles/models/","")
    model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=1000,
              verbose=1,
              tensorboard_log=tb_log,
    )
    # Train the agent
    model.learn(total_timesteps=n_steps)
    # # Save the agent
    save_model(model_name, model)
    # save_stored_training_data(buffer_list, env_name, n_steps)

    if run_afterwards:
        input("press")
        model = load_model(model_name,"dqn")
        env_kwargs["render_mode"] = "human"
        env = create_hw_env([],store_data=False,**env_kwargs)
        env.render()
        for _ in range(5):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                env.render()

if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    main(env_name, steps, run_afterwards=False)
