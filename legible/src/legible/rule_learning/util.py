import os
import pathlib
import pickle

import numpy as np
import torch

from stable_baselines3 import PPO, DQN

def load_pickle(file_name, exact_match = False):
    effective_name = find_file_name(file_name,"pkl",next=False) if not exact_match else file_name
    print(f"Loading {effective_name}")
    if pathlib.Path(effective_name).exists():
        with open(effective_name, "rb") as fp:
            return pickle.load(fp)
    return None


def save_pickle(file_name,data, exact_match = False):
    effective_name = find_file_name(file_name,"pkl",next=True) if not exact_match else file_name
    print(f"Saving {effective_name}")
    from pathlib import Path
    eff_path = Path(effective_name)
    if not eff_path.parent.exists():
        eff_path.parent.mkdir(parents=True)
    with open(effective_name, "wb") as fp:
        pickle.dump(data, fp)

def load_ppo_model(model_name, exact_match = False, env = None):
    effective_name = find_file_name(model_name,"zip",next=False) if not exact_match else model_name
    print(f"Loading {effective_name}")
    if env is not None:
        return PPO.load(effective_name,env)
    else:
        return PPO.load(effective_name)

def save_model(model_name, model, exact_match = False):
    effective_name = find_file_name(model_name,"zip",next=True) if not exact_match else model_name
    print(f"Saving {effective_name}")
    model.save(effective_name)

def find_file_name(name,suffix,next=False):
    i = 1
    while os.path.isfile(f'{name}_{i}.{suffix}'):
        i+=1
    if i == 1 and next == False:
        raise Exception(f"File of the form {name}_{i}.{suffix} does not exist.")
    if next == False:
        return f"{name}_{i-1}.{suffix}" # return name of existing file
    else:
        return f"{name}_{i}.{suffix}"

def find_dir_name(name):
    i = 1
    while os.path.isdir(f'{name}_{i}'):
        i+=1
    return f"{name}_{i}"


def predict_act(model, obs, action_tensor, sample_reconstruction,algo_name):
    batch_size = obs.shape[0]
    obs = sample_reconstruction(obs,batch_size)
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)

    # action_tensor = action_tensor.repeat(batch_size,1)# TODO I dont know why this doesnt work
    act_probs = []
    for i in range(batch_size):
        obs_curr = obs_t[i, :].unsqueeze(0)
        if algo_name == "ppo":
            act_logits = policy.get_distribution(obs_curr).log_prob(action_tensor)
        elif algo_name == "dqn":
            act_logits = policy.q_net(obs_curr).squeeze() # not really act_logits but treat them as a

        else:
            raise Exception("Unsupported algorithm")
        act_probs.append(torch.softmax(act_logits, 0).detach().cpu().numpy())
    return np.array(act_probs)


def load_dqn_model(model_name, exact_match = False, env = None):
    effective_name = find_file_name(model_name, "zip", next=False) if not exact_match else model_name
    print(f"Loading {effective_name}")
    if env is not None:
        return DQN.load(effective_name, env)
    else:
        return DQN.load(effective_name)


def load_model(model_path,algo_name, exact_match = False,env= None):
    if algo_name == "ppo":
        return load_ppo_model(model_path,exact_match = exact_match)
    elif algo_name == "dqn":
        return load_dqn_model(model_path, exact_match=exact_match,env=env)
    else:
        raise Exception("Unsupported")
