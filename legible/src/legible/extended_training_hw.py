from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from hw.hw_util import env_kwargs
from rule_learning.util import save_model, find_file_name
from train_hw import create_env


def ext_train(env_name,base_model_name,n_steps,exact_model_number=None,tb_name = None):

    env_kwargs["id"] = env_name
    env = make_vec_env(
        create_env,
        n_envs=8,
        seed=0,
        env_kwargs=env_kwargs,
    )
    policy_kwargs_dqn = dict(net_arch=[256, 256])

    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None

    if exact_model_number is None:
        base_model_name = find_file_name(base_model_name, suffix="zip")
    else:
        base_model_name = f"{base_model_name}_{exact_model_number}.zip"
        print(f"Going to load model parameters from: {base_model_name}")

    model = DQN("MlpPolicy", env,
                    policy_kwargs=policy_kwargs_dqn,
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.8,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=1000,
                    verbose=1,
                    tensorboard_log=tb_path
                    )
    model.set_parameters(base_model_name)
    model.learn(n_steps)

    return model

def ext_train_hw(env_name, orig_steps,steps,
                 exact_model_number=None
                     ):
    algo = "ext_dqn"
    base_algo = "dqn"
    base_model_name = f"pickles/models/{base_algo}_{env_name.replace('/','_')}_{orig_steps}"
    model_name = f"pickles/models/{algo}_{env_name.replace('/','_')}_{orig_steps}_to_{steps}"
    tb_name = model_name.replace("pickles/models/","")
    model = ext_train(env_name,base_model_name, steps,exact_model_number=exact_model_number,tb_name = tb_name)

    if exact_model_number is None:
        save_model(model_name,model)
    else:
        model_name = f"{model_name}_{exact_model_number}.zip"
        save_model(model_name, model, exact_match=True)



if __name__ == "__main__":
    import sys
    env_name = sys.argv[1]
    orig_steps = int(sys.argv[2])
    steps = int(sys.argv[3])
    shield_rule = None

    exact_model_number = None
    for arg in sys.argv:
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))

    ext_train_hw(env_name,orig_steps, steps, exact_model_number=exact_model_number)