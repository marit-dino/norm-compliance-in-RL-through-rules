import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from rule_learning.util import save_model, find_file_name
from train_pacman import create_pacman_env, run_trained_model


def ext_train(env_name,base_model_name, feature_extractor, n_steps, level, exact_model_number=None,
              tb_name = None):
    env = make_vec_env(lambda : create_pacman_env(env_name, feature_extractor = feature_extractor, level=level, render_mode="none"
                                                  ), 8)

    policy_kwargs_dqn = dict(
        net_arch=[256]*2,
        activation_fn=torch.nn.ReLU,
    )

    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None

    if exact_model_number is None:
        base_model_name = find_file_name(base_model_name, suffix="zip")
    else:
        base_model_name = f"{base_model_name}_{exact_model_number}.zip"
        print(f"Going to load model parameters from: {base_model_name}")


    model = DQN("MlpPolicy", env, verbose =1, tensorboard_log=tb_path, policy_kwargs=policy_kwargs_dqn,
                batch_size=256, buffer_size=50_000, exploration_fraction=0.5, gamma = 0.95,gradient_steps=-1
               )
    model.set_parameters(base_model_name)
    model.learn(n_steps)

    return model

def ext_train_pacman(env_name, orig_steps,steps, level,feature_extractor,exact_model_number=None,
                     run_afterwards=False):

    algo = "ext_dqn"
    base_algo = "dqn"
    base_model_name = f"pickles/models/{base_algo}_{env_name.replace('/','_')}_{orig_steps}_level_{level}_{feature_extractor}"
    model_name = f"pickles/models/{algo}_{env_name.replace('/','_')}_{orig_steps}_to_{steps}_level_{level}_{feature_extractor}"
    tb_name = f"{algo}_{env_name}_mode_{level}"

    model = ext_train(env_name,base_model_name, feature_extractor, steps,level,
                      exact_model_number=exact_model_number,
                      tb_name = tb_name)

    if exact_model_number is None:
        save_model(model_name,model)
    else:
        model_name = f"{model_name}_{exact_model_number}.zip"
        save_model(model_name, model, exact_match=True)
    if run_afterwards:
        input("press")
        run_trained_model(env_name,model,feature_extractor = feature_extractor, level=level)

if __name__ == "__main__":
    import sys
    env_name = sys.argv[1]
    orig_steps = int(sys.argv[2])
    steps = int(sys.argv[3])
    level = sys.argv[4]
    feature_extractor = "extended-8"

    exact_model_number = None
    for arg in sys.argv:

        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))

    ext_train_pacman(env_name,orig_steps, steps, level,feature_extractor,
                     exact_model_number=exact_model_number)