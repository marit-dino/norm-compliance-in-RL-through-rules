import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from mg.env_utils import create_mg_env
from rule_learning.util import save_model, load_ppo_model, find_dir_name


def train_ppo(model, env_name,n_steps, tb_name = None):

    env = make_vec_env(lambda : create_mg_env(env_name), 8)

    # set up logger
    tb_path = f"./tb_log/{tb_name}/PPO" if tb_name is not None else None
    if tb_path:
        tb_path = find_dir_name(tb_path)

    new_logger = configure(tb_path, ["stdout", "tensorboard"])

    model.set_env(env)
    model.set_logger(new_logger)
    model.learn(n_steps)
    return model

def run_trained_model(env_name,model):
    env = create_mg_env(env_name,render_mode="human")

    obs,info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term,trunc, info = env.step(action)
        env.render()
        if term or trunc:
            obs, info = env.reset()


def main(env_name,base_steps, steps,base_env,run_afterwards=False):
    base_model_name = f"pickles/models/ppo_{base_env.replace('/','_')}_{base_steps}"
    ext_model_name = (f"pickles/models/ppo_{env_name.replace('/','_')}_{steps}"
                      f"_ext_from_{base_env}_{base_steps}")
    tb_name = f"ppo_{env_name}_ext_from_{base_env}_{base_steps}"

    print(f"Loading {base_model_name}")
    base_model = load_ppo_model(base_model_name)
    model = train_ppo(base_model, env_name,n_steps=steps, tb_name = tb_name)
    save_model(ext_model_name, model)

    if run_afterwards:
        input("press")
        run_trained_model(env_name,model)

if __name__ == "__main__":
    import sys
    env_name = sys.argv[1]
    base_steps = int(sys.argv[2])
    steps = int(sys.argv[3])
    base_env = sys.argv[4]
    main(env_name,base_steps, steps,base_env)