import sys

import gymnasium.spaces
import numpy as np
import torch
from .rule_learning.data_collection import collect_eps_data_for_rules, get_obs_data_from_eps, extract_based_on_importance, extract_actionwise_min_q
from .rule_learning.feature_detection import detect_features, get_most_important_corr_features
from .rule_learning.learnRules import extract_features_from_obs, get_rules
from .rule_learning.util import save_pickle, load_model
from legible.env_util import create_environment_and_modelname, create_environment_and_modelname_for_oftendeeprl

from shield.shields import AspShield


def reconstruct_atari_obs(ram_obs):
    ram_obs = (ram_obs * 255).astype(np.uint8)
    obs_bits = np.unpackbits(ram_obs,axis=1)
    return np.concatenate((obs_bits, ram_obs / 255),axis=1)


def extract_based_on_focus(all_obs_data, feature_focus, nr_experiences):
    obs_data = []
    for obs in all_obs_data:
        state = obs[1:].astype(np.float32)
        if feature_focus(state):
            obs_data.append(obs)
    print(f"Found {len(obs_data)} data points with focus")
    return np.array(obs_data)


def extract_actionwise_min_prob(obs_data, model, n_actions,action_tensor):
    obs_index_for_action = {k: [] for k in range(n_actions)}
    for row in range(obs_data.shape[0]):
        state = obs_data[row][1:].astype(np.float32)
        obs_t, _vectorized_env = model.policy.obs_to_tensor(state)
        _estimated_value, act_logits, _ = model.policy.evaluate_actions(obs_t, action_tensor)
        min_prob_act = torch.argmin(act_logits).item()
        obs_index_for_action[min_prob_act].append(row)

    obs_index_for_action = {k : np.array(v) for k,v in obs_index_for_action.items()}
    return obs_index_for_action


def dqn_min_q_selection(obs_data, model, num_actions, action_tensor,sample_reconstruction=None):
    obs_index_for_action = {k: [] for k in range(num_actions)}
    for row in range(obs_data.shape[0]):
        state = obs_data[row][1:].astype(np.float32)
        if sample_reconstruction is not None:
            state = sample_reconstruction(state, 1)
        obs_t, _vectorized_env = model.policy.obs_to_tensor(state)
        q_values = model.policy.q_net(obs_t)
        # for some reason, the shape of q values keeps changing, so let's protect against erroneous results and fixed
        # the issue when it occurs again
        assert list(q_values.shape) == [1,action_tensor.shape[0]]
        q_values = q_values.squeeze()
        min_prob_act = torch.argmin(q_values).item()
        obs_index_for_action[min_prob_act].append(row)

    obs_index_for_action = {k: np.array(v) for k, v in obs_index_for_action.items()}
    return obs_index_for_action


def select_features_and_learn_rules(env_name, steps_initial, mode, nr_eps, nr_features, lime_test_size, corr, algo_name,
                                    feature_extractor_deeprl, exact_model_number = None,
                                    extract_min_prob_neg_data = True,
                                    steps_norm = 0,
                                    norm_descriptor = "",
                                    feature_extractor_rules = "",
                                    algorithm="ripper", min_acc=0.9, min_cov=0.01):

    # setup stuff
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # constant settings
    nr_exp_factor = 5
    failure_neighborhood = -1

    if "norm_guided" in algo_name:
        env, model_name, model_path = create_environment_and_modelname_for_oftendeeprl(algo_name, env_name, mode, norm_descriptor, steps_initial, steps_norm,
                                                                   feature_extractor=feature_extractor_deeprl)
    else:
        env, model_name, model_path = create_environment_and_modelname(algo_name, env_name, mode, steps_initial,
                                                                   feature_extractor=feature_extractor_deeprl)
    categorical_features, failure_indicator, nr_features_all, sample_reconstruction,groups_of_similar = (
        get_features_and_failure_indication(env_name, mode,feature_extractor_rules))
    print(f"Loading {model_name}")
    if exact_model_number is None:
        model = load_model(model_path, algo_name)
    else:
        model_path = f"{model_path}_{exact_model_number}"
        print(model_path)
        model = load_model(model_path, algo_name, exact_match=True)

    assert type(env.action_space) == gymnasium.spaces.Discrete
    num_actions = env.action_space.n
    actions_tensor = torch.tensor(range(num_actions), device=device)

    # data collection
    eps_data = collect_eps_data_for_rules(nr_eps, env, model,algo_name, actions_tensor,model_name,
                                          try_load=exact_model_number is None)
    nr_experiences = int(sum(len(d.obs) for d in eps_data) / nr_exp_factor)
    print(f"Going to select {nr_experiences} experiences")
    all_obs_data, eps_data_array,_ = get_obs_data_from_eps(eps_data, nr_experiences,
                                                         return_all=True)
    all_obs_data_failure, eps_data_array_failure,_ = get_obs_data_from_eps(eps_data, nr_experiences,
                                                                           failure_neighborhood=failure_neighborhood,
                                                                           failure_indicator=failure_indicator,
                                                         return_all=True)

    # reselect for failure focus
    if failure_neighborhood > 0:
        eps_data_array = eps_data_array_failure
        all_obs_data = all_obs_data_failure

    features, feature_intervals, feature_importances = (
        detect_features(env_name,eps_data_array, model, nr_features, lime_test_size, actions_tensor, nr_features_all,
                            num_actions, algo_name, groups_of_similar=groups_of_similar,
                        categorical_features=categorical_features, compute_correlation=corr,
                        only_features_for_neg=False, sample_reconstruction = sample_reconstruction))
    
    feature_indices, feature_names = zip(*features)
    feature_indices = list(feature_indices)
    feature_names = list(feature_names)
    feature_focus = None # lambda state : state[19] <= 2
    print("Detected features")
    X, target = extract_features_from_obs(eps_data_array, feature_indices, feature_intervals, categorical_features)
    # X is of shape #samples x #extracted features
    feature_indices_reduced = get_most_important_corr_features(X, target, feature_indices)

    # rule stuff
    eps_data_array_reselected, obs_data_min_q_index = (
        reselect_data(actions_tensor, algo_name, all_obs_data, eps_data_array, extract_min_prob_neg_data, feature_focus,
                  model, nr_experiences, sample_reconstruction))

    X, target = extract_features_from_obs(eps_data_array_reselected, feature_indices, feature_intervals, categorical_features)
    print("Learning neg_rules")
    neg_rules = get_rules(X[[f'f{var}' for var in feature_indices_reduced]], target, feature_importances,
                      actions=list(range(num_actions)),obs_data_min_q_index=obs_data_min_q_index,
                      algorithm=algorithm, MIN_ACC=0.9, MIN_COV=min_cov,negated=True)
    print("Learned neg_rules")
    if len(neg_rules)==0:
        raise Exception("No negative rules learned")
        
    for rule in neg_rules:
        print(f'{rule[0]} \t accuracy = {rule[1][0]}, coverage = {rule[1][1]}')

    neg_rules = "\n".join(r[0] + "." for r in neg_rules)

    print("Learning pos_rules")
    pos_rules = get_rules(X[[f'f{var}' for var in feature_indices_reduced]], target, feature_importances,
                      actions=list(range(num_actions)),
                      algorithm="ripper", MIN_ACC=min_acc, MIN_COV=min_cov,negated=False)


    for rule in pos_rules:
        print(f'{rule[0]} \t accuracy = {rule[1][0]}, coverage = {rule[1][1]}')

    pos_rules = "\n".join(r[0] + "." for r in pos_rules)
    print(pos_rules)

    shield = AspShield(num_actions, feature_indices_reduced, feature_names, neg_rules, pos_rules,
                       feature_intervals, categorical_features)

    shield_name = f"pickles/shields/{'corr'if corr else 'uncorr'}/"\
                  f"{algo_name}_{env_name.replace('/','_')}_{mode}_feat_{nr_features}_{steps_initial}_to_{steps_norm}_shield"

    if exact_model_number is None:
        save_pickle(shield_name,shield)
    else:
        shield_name = f"{shield_name}_{exact_model_number}.pkl"
        save_pickle(shield_name,shield,exact_match=True)

    return shield,shield_name


def reselect_data(actions_tensor, algo_name, all_obs_data, eps_data_array, extract_min_prob_neg_data, feature_focus,
                  model, nr_experiences, sample_reconstruction = None):
    num_actions = actions_tensor.shape[0]
    obs_data_min_q_index = None
    if feature_focus is not None:
        eps_data_array_reselected = extract_based_on_focus(all_obs_data, feature_focus, nr_experiences)
    else:
        eps_data_array_reselected = eps_data_array
    if algo_name == "ppo" and extract_min_prob_neg_data:
        eps_data_array_reselected = eps_data_array
        obs_data_min_q_index = extract_actionwise_min_prob(eps_data_array_reselected, model, num_actions,
                                                           actions_tensor)
    elif "dqn" in algo_name and extract_min_prob_neg_data:
        obs_data_min_q_index = dqn_min_q_selection(eps_data_array_reselected, model, num_actions, actions_tensor,sample_reconstruction)
    return eps_data_array_reselected, obs_data_min_q_index


def get_features_and_failure_indication(env_name,mode, feature_extractor):
    failure_indicator = None
    groups_of_similar = []
    if "MiniGrid" in env_name:
        nr_features_all = 7 * 7 * 3 + 1
        categorical_features = list(range(7 * 7 * 3))
        sample_reconstruction = None
    elif "highway" in env_name or "intersection" in env_name or "roundabout" in env_name or "merge" in env_name:
        print("Setting up a highway-env")
        nr_features_all = 7 * 10
        categorical_features = []
        sample_reconstruction = lambda obs, batch_size: obs.reshape(batch_size, 10, 7)
        groups_of_similar = []
        for i in range(7):
            sim_feat_from_i = list(range(i,70,7))
            groups_of_similar.append(sim_feat_from_i)
    else:
        sample_reconstruction = None
        if "original" in mode:
            nr_ghosts = 4
        elif "small" in mode or "medium" in mode:
            nr_ghosts = 2
        else:
            raise Exception("Unknown env.")
        nr_features_all = 21 + nr_ghosts * 24
        if feature_extractor == "extended-9":
            nr_features_all += 8
        categorical_features = list(range(nr_features_all))
        if nr_ghosts == 2 or nr_ghosts == 4:
            categorical_features.remove(61)
            categorical_features.remove(60)
            categorical_features.remove(59)
            categorical_features.remove(52)
            categorical_features.remove(37)
            categorical_features.remove(36)
            categorical_features.remove(35)
            categorical_features.remove(28)
            categorical_features.remove(8)
            categorical_features.remove(7)
            categorical_features.remove(1)
            categorical_features.remove(0)
        if nr_ghosts == 2:
            categorical_features.remove(68)
            categorical_features.remove(67)
        elif nr_ghosts == 4:
            categorical_features.remove(76)
            categorical_features.remove(83)
            categorical_features.remove(84)
            categorical_features.remove(85)
            categorical_features.remove(100)
            categorical_features.remove(107)
            categorical_features.remove(108)
            categorical_features.remove(109)
            categorical_features.remove(115)
            categorical_features.remove(116)

        failure_indicator = lambda obs, r: r < -0.9
    return categorical_features, failure_indicator, nr_features_all, sample_reconstruction,groups_of_similar


if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    mode = int(sys.argv[3]) if "Pacman" not in env_name else sys.argv[3]
    nr_eps = int(sys.argv[4])
    nr_features = int(sys.argv[5])
    corr = sys.argv[6] == "True"
    lime_test_size = 100
    algo = "dqn"

    feature_extractor = "extended-8"
    exact_model_number = None
    for arg in sys.argv:
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))

    select_features_and_learn_rules(env_name, steps, mode, nr_eps, nr_features, lime_test_size, corr, algo,
                                    feature_extractor, exact_model_number = exact_model_number)
