import random
from dataclasses import dataclass

import numpy as np
import torch
import torch as tc
from stable_baselines3.common.buffers import RolloutBuffer

from rule_learning.util import save_pickle, load_pickle

from torch.nn import functional as F

@dataclass
class EpisodeData:
    obs : list
    act_logits : list
    est_values : list
    actions : list
    rewards : list


def generate_episode_data(env, model, algo_name, action_tensor) -> EpisodeData:
    obs, info = env.reset()
    policy = model.policy
    obs_t, vectorized_env = policy.obs_to_tensor(obs)
    obs_t = obs_t.to(action_tensor.device)
    if algo_name == "ppo":
        estimated_value, act_logits,_ = policy.evaluate_actions(obs_t,action_tensor) #.log_prob(action_tensor)
    elif "dqn" in algo_name:
        q_values = policy.q_net(obs_t)
        act_logits = q_values # not logits but treat them as such for now
        estimated_value = torch.max(act_logits)
    else:
        raise Exception("Unsupported")

    data = EpisodeData([],[], [],[],[])
    data.obs.append(obs)
    data.act_logits.append(act_logits)
    data.est_values.append(estimated_value)
    while True:
        action, _states = model.predict(obs)
        obs, reward, term, trunc, info = env.step(action)

        obs_t, vectorized_env = policy.obs_to_tensor(obs)
        obs_t = obs_t.to(action_tensor.device)

        if algo_name == "ppo":
            estimated_value, act_logits, _ = policy.evaluate_actions(obs_t, action_tensor)  # .log_prob(action_tensor)
        elif "dqn" in algo_name:
            q_values = policy.q_net(obs_t)
            act_logits = q_values  # not logits but treat them as such for now
            estimated_value = torch.max(act_logits)
        else:
            raise Exception("Unsupported")
        data.est_values.append(estimated_value)
        data.act_logits.append(act_logits)
        data.actions.append(action)
        data.obs.append(obs)
        if term and reward <= 0:
            data.rewards.append(min(-1,reward))
        else:
            data.rewards.append(reward)
        if term or trunc:
            break

    return data

def collect_eps_data_for_rules(nr_eps,env,model,algo_name,action_tensor,model_name,try_load=True, relearn=False):
    print(f"Going to collect {nr_eps} episodes")
    pickle_f_name = f"pickles/episodes/{model_name}_{nr_eps}{'_relearn' if relearn else ''}"
    if try_load:
        try:
            eps_data = load_pickle(pickle_f_name)
            if eps_data is not None:
                return eps_data
        except Exception:
            pass
    else:
        print("We don't try to load episode because use exact model numbers.")
    with torch.no_grad():
        eps_data = []
        for i in range(nr_eps):
            print(f"Episode {i}")
            episode_data = generate_episode_data(env,model,algo_name,action_tensor)
            eps_data.append(episode_data)
        save_pickle(pickle_f_name,eps_data)
    return eps_data

def collect_exp_data_for_rules(nr_exp,env,model,model_name,relearn=False):
    print(f"Going to collect {nr_exp} experiences")
    pickle_f_name = f"pickles/rollout/{model_name}_{nr_exp}{'_relearn' if relearn else ''}"
    try:
        eps_data = load_pickle(pickle_f_name)
        if eps_data is not None:
            return eps_data
    except Exception:
        pass
    with torch.no_grad():
        env.reset()
        rollout_buffer = RolloutBuffer(nr_exp,env.observation_space,env.action_space,n_envs=8)
        model.collect_rollouts(env, model._init_callback(None) , rollout_buffer, n_rollout_steps=nr_exp)
        save_pickle(pickle_f_name,rollout_buffer)
    return rollout_buffer


def get_obs_data_from_rollout_buffer(rollout_buffer, model, nr_data_points,normalize_advantage=True,clip_range_vf=None):
    # copied and adapted from PPO train (parameters are defaults from PPO)
    data = []
    for rollout_data in rollout_buffer.get(1):
        # Convert discrete action from float to long
        actions = rollout_data.actions.long().flatten()

        values, log_prob, entropy = model.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the difference between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + tc.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values_pred)
        row = np.concatenate((np.array([actions[0].cpu().numpy()]), rollout_data.observations[0].cpu().numpy()))

        data.append((row,value_loss.item()))

    data.sort(key=lambda x: x[1])
    data = [r for r,v in data[:nr_data_points]]
    return np.array(data)


def has_failing_state(e,failure_indicator):
    return any([failure_indicator(obs_s, reward_s)
                for (obs_s,reward_s) in zip(e.obs, e.rewards)])


def filter_eps_for_failing(eps_data,failure_indicator):
    return [e for e in eps_data if has_failing_state(e,failure_indicator)]


def is_in_failure_neighborhood(pos, single_eps_data, neighborhood_size, failure_indicator):
    neighborhood = list(zip(single_eps_data.obs,single_eps_data.rewards))[pos:pos+neighborhood_size]
    return any([failure_indicator(obs_s, reward_s)
                for (obs_s,reward_s) in neighborhood])


def get_obs_data_from_eps(eps_data, nr_data_points, failure_neighborhood = -1,
                          failure_indicator = None,
                          act_probs_for_label=False,
                          return_all=False):
    assert (not (failure_neighborhood > 0)) or failure_indicator is not None
    obs_data = []
    act_logit_data = []
    if failure_neighborhood > 0:
        eps_data = filter_eps_for_failing(eps_data,failure_indicator)
        print(f"{len(eps_data)}")
    for e in eps_data:
        for i in range(len(e.obs)):
            if (failure_neighborhood <= 0) or (failure_neighborhood > 0 and is_in_failure_neighborhood(i,e,
                                                                            failure_neighborhood,failure_indicator)):
                act_logits = e.act_logits[i]
                act_logit_data.append(act_logits.cpu().numpy())
                obs = e.obs[i]
                obs_flat = obs.flatten()
                value = e.est_values[i] if isinstance(e.est_values[i],float) else e.est_values[i].item()
                if act_probs_for_label:
                    label = tc.softmax(act_logits,0)
                    row = np.concatenate((label.cpu().numpy(), obs_flat))
                else:
                    label = tc.argmax(act_logits)
                    row = np.concatenate((np.array([label.data.cpu().numpy()]),obs_flat))

                obs_data.append((row, value))
    all_obs_data = [r for r,v in obs_data]
    if failure_neighborhood > 0:
        print(f"We have {len(all_obs_data)} data points in the neighborhood of failures")

    if nr_data_points < len(obs_data):
        choice_indices = random.choices(range(len(obs_data)),k=nr_data_points)
    else:
        choice_indices = range(len(obs_data))
    obs_data = [obs_data[i][0] for i in choice_indices]
    act_logit_data = [act_logit_data[i] for i in choice_indices]
    if return_all:
        return all_obs_data,np.array(obs_data),np.array(act_logit_data)
    else:
        return np.array(obs_data)

def extract_based_on_importance(all_obs_data, q_val_calculator, nr_data_points):
    obs_data_with_imp = []
    for obs in all_obs_data:
        state = obs[1:].astype(np.float32)
        imp = q_val_calculator.compute_importance(state)
        obs_data_with_imp.append((obs,imp))

    obs_data_with_imp.sort(key=lambda x: x[1], reverse=True)

    imp_threshold = obs_data_with_imp[nr_data_points-1][1]
    print(f"Threshold {imp_threshold}")
    obs_data = [r for r,v in obs_data_with_imp[:nr_data_points]]
    return np.array(obs_data),imp_threshold

def extract_actionwise_min_q(obs_data, q_val_calculator):
    obs_index_for_action = {k : [] for k in range(q_val_calculator.n_actions)}
    for row in range(obs_data.shape[0]):
        state = obs_data[row].astype(np.float32)
        min_action = q_val_calculator.compute_min_q_action(state)
        obs_index_for_action[min_action].append(row)
    obs_index_for_action = {k : np.array(v) for k,v in obs_index_for_action.items()}

    return obs_index_for_action