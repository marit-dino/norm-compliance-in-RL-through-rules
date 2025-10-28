import copy
import hashlib
import random
import sys
import warnings
from collections import defaultdict
from random import choice
from typing import Any, ClassVar, Optional, TypeVar, Union, Dict, List, Type

import gymnasium
import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, RolloutReturn, \
    TrainFrequencyUnit, ReplayBufferSamples, PyTorchObs, DictReplayBufferSamples
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update, \
    should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from stable_baselines3.dqn.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.dqn.policies import QNetwork, DQNPolicy
from torch import nn

from torch.nn import functional as F
from torch.optim import Adam

from sb3_ext.clingoHelper import ClingoHelper
SelfRuleDQfD = TypeVar("SelfRuleDQfD", bound="RuleDQfD")


class DictReplayBufferWithIgnore(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,handle_timeout_termination=handle_timeout_termination)
        self.ignore = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(  # type: ignore[override]
            self,
            obs: dict[str, np.ndarray],
            next_obs: dict[str, np.ndarray],
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: list[dict[str, Any]],
    ) -> None:
        self.ignore[self.pos] = np.array([1 if info["ignore"] else 0  for info in infos])
        super().add(obs,next_obs,action,reward,done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        batch_inds, env_indices = sample_indices(batch_size,
                                                 self.full, self.buffer_size, self.pos, self.n_envs,
                                                 self.optimize_memory_usage, self.ignore)
        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()},
                                   env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(
                self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )

class ReplayBufferWithIgnore(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,handle_timeout_termination=handle_timeout_termination)
        self.ignore = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        self.ignore[self.pos] = np.array([1 if info["ignore"] else 0  for info in infos])
        # self.ignore[self.pos] = np.array([0 for info in infos]) #np.array([1 if info["ignore"] else 0  for info in infos])
        super().add(obs,next_obs,action,reward,done, infos)


    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        batch_inds, env_indices = sample_indices(batch_size,
                                                 self.full, self.buffer_size, self.pos, self.n_envs,
                                                 self.optimize_memory_usage, self.ignore)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1,
                                                                                                         1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


def sample_indices(batch_size, full, buffer_size, pos, n_envs, optimize_memory_usage, ignore):
    if full:
        batch_inds = (np.random.randint(1, buffer_size, size=batch_size * 4) + pos) % buffer_size
    else:
        batch_inds = np.random.randint(0, pos, size=batch_size * 4)
    env_indices = np.random.randint(0, high=n_envs, size=(len(batch_inds),))
    b_indexes = []
    e_indexes = []
    for bi_index in range(batch_inds.shape[0]):
        for ei_index in range(env_indices.shape[0]):
            bi = batch_inds[bi_index]
            ei = env_indices[ei_index]
            if optimize_memory_usage:
                do_ignore = ignore[(bi + 1) % buffer_size, ei]
            else:
                do_ignore = ignore[bi, ei]
            if not do_ignore:
                b_indexes.append(bi)
                e_indexes.append(ei)
            if len(b_indexes) >= batch_size:
                break
        if len(b_indexes) >= batch_size:
            break
    while len(b_indexes) < batch_size:
        bi = np.random.randint(1, buffer_size) if full else np.random.randint(0, pos)
        ei = np.random.randint(0, n_envs)
        if optimize_memory_usage:
            do_ignore = ignore[(bi + 1) % buffer_size, ei]
        else:
            do_ignore = ignore[bi, ei]
        if not do_ignore:
            b_indexes.append(bi)
            e_indexes.append(ei)
    batch_inds = np.array(b_indexes)
    env_indices = np.array(e_indexes)
    return batch_inds, env_indices


class RuleDQfD(OffPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        clingo_helper : ClingoHelper,
        gym_to_action_names,
        final_shield_rate,
        norm_violation_filtering=True,
        add_expert_loss=True,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        margin : int = 50,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        # self.triggered_obs = set()
        np.set_printoptions(threshold=sys.maxsize)
        self.last_rules_triggered = None
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        self.norm_violation_filtering = norm_violation_filtering
        self.add_expert_loss = add_expert_loss


        self.meta_rule_mode = False
        self.shield_rate = 0.1
        self.final_shield_rate = final_shield_rate
        self.large_number = 1e6
        self.action_list = list(range(self.action_space.n))
        self.nr_actions = self.action_space.n
        self.margin = margin
        replay_buffer_class = DictReplayBufferWithIgnore \
            if type(env.observation_space) == gymnasium.spaces.Dict else ReplayBufferWithIgnore

        self.rule_replay_buffer = replay_buffer_class(self.buffer_size,self.observation_space,self.action_space,
                                                           self.device,self.n_envs)
        self.replay_buffer = replay_buffer_class(self.buffer_size,self.observation_space,self.action_space,
                                                           self.device,self.n_envs)
        # monitor_wrapper is set True in super
        self.rule_env = self.make_env_copy(env,policy,monitor_wrapper=True)

        self.no_rule_expl = True
        # shield_rate not necessary
        self.use_shield_rate = False
        self.norm_helper = clingo_helper
        if _init_setup_model:
            self._setup_model()
        self.gym_to_action_names = gym_to_action_names
        self.action_names_to_int = {a : i for i,a in self.gym_to_action_names.items()}


    def get_state(self,env : VecEnv):
        try:
            state_list = env.env_method("get_state")
        except: # SUMO-RL does not have get_state but __getstate__
            state_list = [{'env' : env.envs[0].env}]
            # state_list = env.env_method("__getstate__")
        return state_list

    def clean_up(self):
        del self.norm_helper


    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "q_net", "q_net_target","clingo_helper"]

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self.shield_schedule = get_linear_fn(
            0.1,
            self.final_shield_rate,
            1,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )


    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super()._store_transition(replay_buffer,buffer_action,new_obs,reward, dones, infos)


    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.shield_rate = self.shield_schedule(self._current_progress_remaining)

        self.logger.record("rollout/exploration_rate", self.exploration_rate)


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            rule_replay_data = self.rule_replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                # same for rules
                rule_next_q_values = self.q_net_target(rule_replay_data.next_observations)
                rule_next_q_values, _ = rule_next_q_values.max(dim=1)
                rule_next_q_values = rule_next_q_values.reshape(-1, 1)
                rule_target_q_values = rule_replay_data.rewards + (1 - rule_replay_data.dones) * self.gamma * rule_next_q_values

                rule_current_q_exp_loss = self.q_net_target(rule_replay_data.observations)
                rule_current_q_exp_loss = th.gather(rule_current_q_exp_loss, dim=1, index=rule_replay_data.actions.long())

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # same for rules
            rule_current_q_values = self.q_net(rule_replay_data.observations)
            rule_current_q_values = th.gather(rule_current_q_values, dim=1, index=rule_replay_data.actions.long())
            loss += F.smooth_l1_loss(rule_current_q_values, rule_target_q_values)

            # now the expert loss

            if self.add_expert_loss:
                rule_current_q_values = self.q_net(rule_replay_data.observations)  # batch size x actions
                margins = (torch.ones(len(self.action_list), len(self.action_list)) -
                           torch.eye(len(self.action_list))) * self.margin
                state_margins = rule_current_q_values + margins.to(self.device)[rule_replay_data.actions.squeeze()]

                supervised_loss = F.l1_loss(state_margins.max(1)[0].unsqueeze(1),rule_current_q_exp_loss)
                # rule_target_q_values,_ = (rule_current_q_values + margin_function).max(dim=1)
                # loss += (1-self.shield_rate)*F.smooth_l1_loss(rule_current_q_exp_loss, rule_target_q_values)
                loss += self.shield_rate * supervised_loss
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            # self.update_lagrange_multiplier(rule_loss.item())
        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/shield_rate", self.shield_rate)


    # copied and adapted from super
    def learn(
            self: SelfRuleDQfD,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfRuleDQfD:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None, "You must set the environment before calling learn()"
        assert isinstance(self.train_freq, TrainFreq)  # check done in _setup_learn()

        while self.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts_norm(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            # print(f"Collected {rollout.episode_timesteps} episode time steps in normal mode")

            # don't want to store expert infos
            ep_info_buffer_save = copy.deepcopy(self.ep_info_buffer)
            ep_success_buffer_save = copy.deepcopy(self.ep_success_buffer)
            ep_num_save = self._episode_num
            timestep_save = self.num_timesteps
            self.rule_mode = True

            rollout = self.collect_rollouts_norm(
                self.rule_env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.rule_replay_buffer,
                log_interval=None,
            )

            # print(f"Collected {rollout.episode_timesteps} episode time steps in rule mode")
            self.rule_mode = False
            self.ep_info_buffer = ep_info_buffer_save
            self.ep_success_buffer = ep_success_buffer_save
            self._episode_num = ep_num_save
            self.num_timesteps = timestep_save

            if not rollout.continue_training:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def collect_rollouts_norm(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            self.env_states = self.get_state(env)
            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)
            # if self.num_timesteps < learning_starts:
            #     changed = [False] * env.num_envs
            # else:
            if self.norm_violation_filtering:
                _fix_action, changed = self.enforce_policy_fix(env.num_envs,self._last_obs,actions)
            else:
                changed = [False]*env.num_envs
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            
            # print(sum(changed)/len(changed))
            for i,ch in enumerate(changed):
                infos[i]["ignore"] = ch

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if not callback.on_step():
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)  # type: ignore[arg-type]

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    def predict(
            self,
            observation: Union[np.ndarray, dict[str, np.ndarray]],
            state: Optional[tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if self.rule_mode:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
            else:
                n_batch = 1
            if not deterministic and np.random.rand() < self.exploration_rate:
                if self.policy.is_vectorized_observation(observation):
                    action = np.array([self.action_space.sample() for _ in range(n_batch)])
                else:
                    action = np.array(self.action_space.sample())
            else:
                actions, _changed = self.enforce_policy_fix(n_batch, observation)
                action = np.array(actions)

            return action,state
        else:
            if not deterministic and np.random.rand() < self.exploration_rate:
                if self.policy.is_vectorized_observation(observation):
                    if isinstance(observation, dict):
                        n_batch = observation[next(iter(observation.keys()))].shape[0]
                    else:
                        n_batch = observation.shape[0]
                    action = np.array([self.action_space.sample() for _ in range(n_batch)])
                else:
                    action = np.array(self.action_space.sample())
            else:
                action, state = self.policy.predict(observation, state, episode_start, deterministic)
            return action, state

    def enforce_policy_fix(self, n_batch, observation, chosen_actions = None):
        actions = []
        changed = []
        for i in range(n_batch):
            env_state = self.env_states[i]
            obs_i = observation[i, :]
            fix_action, orig_action = self.policy_fix_single(self.margin,self.gym_to_action_names,self.nr_actions,
                                                             self.policy,self.norm_helper,
                                                             chosen_actions, env_state, i, obs_i)
            if fix_action == "Stop":
                fix_action_index = random.randint(0, 3)
            else:
                fix_action_index = self.action_names_to_int[fix_action]
            if fix_action_index != orig_action:
                changed.append(True)
            else:
                changed.append(False)
            actions.append(fix_action_index)
        return actions,changed

    @staticmethod
    def policy_fix_single(margin,gym_to_action_names,nr_actions,policy,norm_helper, chosen_actions, env_state, i, obs_i):
        relevant_states = norm_helper.get_relevant_states(env_state, obs_i)
        action_value_pairs_dict = dict()
        orig_action = None
        for (rel_s, rel_s_id) in relevant_states:
            if rel_s is not None:
                obs_t, _vectorized = policy.obs_to_tensor(rel_s)
                q_values = policy.q_net(obs_t).squeeze()
                action_value_pairs = []

                orig_action = -1
                max_q = -1e10
                for a in range(nr_actions):
                    act_name = gym_to_action_names[a]
                    q = q_values[a].item()
                    action_value_pairs.append((act_name, q))
                    if q > max_q:
                        max_q = q
                        orig_action = a
                if chosen_actions is not None:
                    for action_int, (a, v) in enumerate(action_value_pairs):
                        if action_int == chosen_actions[i]:
                            action_value_pairs[action_int] = (a, max_q + margin)
                            break
                    orig_action = chosen_actions[i]
            else:
                action_value_pairs = []
                for a in range(nr_actions):
                    act_name = gym_to_action_names[a]
                    action_value_pairs.append((act_name, -1))

            action_value_pairs_dict[rel_s_id] = action_value_pairs
        fix_action = norm_helper.get_action(env_state, action_value_pairs_dict)
        return fix_action, orig_action

    # copied from base class
    def  make_env_copy(self,env,policy, monitor_wrapper, supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,):
        from stable_baselines3.common.vec_env import unwrap_vec_normalize
        from stable_baselines3.common.base_class import maybe_make_env
        if env is not None:
            env = maybe_make_env(env, self.verbose)
            env = self._wrap_env(env, self.verbose, monitor_wrapper)

            if supported_action_spaces is not None:
                assert isinstance(self.action_space, supported_action_spaces), (
                    f"The algorithm only supports {supported_action_spaces} as action spaces "
                    f"but {self.action_space} was provided"
                )

            if not support_multi_env and self.n_envs > 1:
                raise ValueError(
                    "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
                )

            # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
            if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(self.observation_space, spaces.Dict):
                raise ValueError(f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}")

            if self.use_sde and not isinstance(self.action_space, spaces.Box):
                raise ValueError("generalized State-Dependent Exploration (gSDE) can only be used with continuous actions.")

            if isinstance(self.action_space, spaces.Box):
                assert np.all(
                    np.isfinite(np.array([self.action_space.low, self.action_space.high]))
                ), "Continuous action space must have a finite lower and upper bound"
            return env


class DQNAdaptationPolicy(DQNPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Discrete, lr_schedule: Schedule,
                 adaptation_layers: list[int], adaptation_activation : Type[nn.Module]):
        super().__init__(observation_space, action_space, lr_schedule)
        self.original_network = self.q_net
        self.original_network_target = self.q_net_target
        second_to_last_layer_dim = None
        self.adaptation_layers = create_mlp(input_dim=second_to_last_layer_dim,output_dim=int(self.action_space.n),
                                            net_arch=adaptation_layers,activation_fn = adaptation_activation)

        # def create_mlp(
        #         input_dim: int,
        #         output_dim: int,
        #         net_arch: list[int],
        #         activation_fn: type[nn.Module] = nn.ReLU,
        #         squash_output: bool = False,
        #         with_bias: bool = True,
        #         pre_linear_modules: Optional[list[type[nn.Module]]] = None,
        #         post_linear_modules: Optional[list[type[nn.Module]]] = None,
        # ) -> list[nn.Module]:


