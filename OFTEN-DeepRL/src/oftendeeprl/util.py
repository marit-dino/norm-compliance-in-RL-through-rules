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


def save_model(model_name, model, exact_match = False):
    effective_name = find_file_name(model_name,"zip",next=True) if not exact_match else model_name
    print(f"Saving {effective_name}")
    model.save(effective_name)
    return effective_name

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


def load_dqn_model(model_name, exact_match = False, env = None):
    effective_name = find_file_name(model_name, "zip", next=False) if not exact_match else model_name
    print(f"Loading {effective_name}")
    if env is not None:
        return DQN.load(effective_name, env)
    else:
        return DQN.load(effective_name)


def load_model(model_path,algo_name, exact_match = False,env= None):
    if "dqn" in algo_name:
        return load_dqn_model(model_path, exact_match=exact_match,env=env)
    else:
        raise Exception("Unsupported")
