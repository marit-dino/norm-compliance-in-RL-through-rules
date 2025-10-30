from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from .env_util import get_policy_kwargs_dqn, get_gym_to_action_names, get_norm_helper
from .sb3_ext.pacman_helper import PacmanClingoHelper
from .util import save_model, find_file_name
from .sb3_ext.clingoHelper import ClingoHelper
from .sb3_ext.DQfD import RuleDQfD
from .train_pacman import create_pacman_env, run_trained_model

def ext_train(env_name, base_model_name, feature_extractor, n_steps, level, norm_helper,
              exact_model_number=None,
              ng_margin = None,
              norm_violation_filtering = True,
              tb_name = None):

    if "Pacman" in env_name:
        env = make_vec_env(lambda : create_pacman_env(env_name, feature_extractor = feature_extractor, level=level, render_mode="none"), 8)
        policy = "MlpPolicy"
    elif "gardener" in env_name:
        size, pct_walls, pct_plants, pct_frogs, seed = level
        env = make_vec_env(
            lambda: create_gardener_env(size,pct_walls, pct_plants, pct_frogs,9,seed),
            8)
        norm_helper.setup(env.get_attr("instance",[0])[0])
        policy = "MlpPolicy"
    elif "sumo" in env_name:
        intersection_type, ambulance_prob, flow_north_south_prob, flow_west_east_prob, seed = level
        # we only run 1 environment in parallel since libsumo only allows 1.
        # It is faster with libsumo than with environments in parallel.
        # However, we use make_vec_env() to be in line with the other environments
        # We set the seed in the vectorized environment to avoid the seed problem
        # with libsumo
        env = make_vec_env(
            lambda: create_sumo_env(
                net_file="./gym_sumo/experiments/singleIntersection/singleIntersection.net.xml",
                route_file=f"./gym_sumo/experiments/singleIntersection/{intersection_type}-ambulance{ambulance_prob}-flowns{flow_north_south_prob}-flowwe{flow_west_east_prob}.rou.xml",
                out_csv_name=None,
                use_gui=False,
                num_seconds=1000,
                min_green=10,
                max_green=50,
                delta_time=2,
            ), n_envs = 1, seed=int(seed))
        env.reset()
        # clingo_excluded = ['t'] excludes the intersection
        clingo_excluded = []
        clingo_static = SumoHelper.get_clingo_static(env=env.envs[0].env,
                                                       excluded=clingo_excluded)
        norm_helper.setup(clingo_static, clingo_excluded)
        policy = "MlpPolicy"
    else:
        raise Exception("Unsupported")

    policy_kwargs_dqn = get_policy_kwargs_dqn(env_name)
    gym_to_action_names = get_gym_to_action_names(env_name)

    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None

    if base_model_name != "":
        if exact_model_number is None:
            base_model_name = find_file_name(base_model_name, suffix="zip")
        else:
            base_model_name = f"{base_model_name}_{exact_model_number}.zip"
            print(f"Going to load model parameters from: {base_model_name}")
    else:
        print("Going to train from scratch.")

    if norm_helper is not None and ng_margin is not None:
        if "Pacman" in env_name:
            final_shield_rate = 1
        elif "gardener" in env_name:
            final_shield_rate = 0.5
        elif "sumo" in env_name:
            final_shield_rate = 0.5

        add_expert_loss = ng_margin > 0
        print(f"Adding expert: {add_expert_loss} {ng_margin} / Norm violation filtering: {norm_violation_filtering}")
        model = RuleDQfD(policy, env, norm_helper, gym_to_action_names, norm_violation_filtering = norm_violation_filtering,
                         add_expert_loss = add_expert_loss,
                         final_shield_rate = final_shield_rate ,verbose =1,
                         tensorboard_log=tb_path, policy_kwargs=policy_kwargs_dqn,
                         batch_size=256, buffer_size=50_000, exploration_fraction=0.5,
                         gamma = 0.95,gradient_steps=-1, margin = ng_margin
                   )
    else:
        model = DQN(policy, env, verbose =1, tensorboard_log=tb_path, policy_kwargs=policy_kwargs_dqn,
                    batch_size=256, buffer_size=50_000, exploration_fraction=0.5, gamma = 0.95,gradient_steps=-1
                   )
    if base_model_name != "":
        model.set_parameters(base_model_name)
    # Stop training if there is no improvement after more than 20 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=20, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    env.reset()
    # The training will stop as soon as the the number of consecutive evaluations without model
    # improvement is greater than 20
    model.learn(n_steps)#, callback=eval_callback)

    if ng_margin is not None:
        model.clean_up()
    return model



def setup_and_ext_train(env_name, orig_steps, steps, level, feature_extractor, norm_helper,
                        norm_descriptor = None,
                        exact_model_number=None,
                        ng_margin = None, norm_violation_filtering=True,run_afterwards=False):
    if norm_helper is not None and ng_margin is not None:
        algo = "norm_guided_dqn"
    else:
        algo = "ext_dqn"
    base_algo = "dqn"

    if "garden" in env_name:
        model_level_name = get_garden_level_name(level)
        feature_ext_str = ""
    elif "Pacman" in env_name:
        model_level_name = level
        feature_ext_str = f"_{feature_extractor}"
    elif "sumo" in env_name:
        model_level_name = get_sumo_level_name(level)
        feature_ext_str = ""
    else:
        raise Exception("Unsupported")

    if int(orig_steps) == 0:
        base_model_name = ""
    else:
        base_model_name = f"pickles/models/{base_algo}_{env_name.replace('/','_')}_{orig_steps}_level_{model_level_name}{feature_ext_str}"
    model_name = f"pickles/models/{algo}_{env_name.replace('/','_')}_{orig_steps}_to_{steps}_level_{model_level_name}{feature_ext_str}"
    tb_name = f"{algo}_{env_name}_mode_{model_level_name}"

    model = ext_train(env_name, base_model_name, feature_extractor, steps, level, norm_helper,
                      exact_model_number=exact_model_number,
                      ng_margin= ng_margin, norm_violation_filtering=norm_violation_filtering,tb_name = tb_name)
    if "norm_guided_dqn" in algo:
        model_name = model_name.replace("norm_guided_dqn",
                                   f"norm_guided_dqn__{norm_descriptor}_")
        if not norm_violation_filtering:
            model_name = model_name.replace("norm_guided_dqn",
                                            f"norm_guided_dqn_no_filter")
        if ng_margin <= 0:
            model_name = model_name.replace("norm_guided_dqn",
                                            f"norm_guided_dqn_no_exp")
    if exact_model_number is None:
        if "sumo" in model_name:
            del model.rule_env
            del model.env_states
        save_model(model_name,model)
    else:
        model_name = f"{model_name}_{exact_model_number}.zip"
        save_model(model_name, model, exact_match=True)
    if run_afterwards:
        input("press")
        run_trained_model(env_name,model,feature_extractor = feature_extractor, level=level)


def parse_level(env_name, level):
    if "Pacman" in env_name:
        return level
    elif "garden" in env_name:
        split_param = level.split("-")
        size = int(split_param[0])
        pct_walls = float(split_param[1])
        pct_plants = float(split_param[2])
        pct_frogs = float(split_param[3])
        seed = float(split_param[4])
        return size,pct_walls,pct_plants,pct_frogs,seed
    elif "sumo" in env_name:
        split_param = level.split("-")
        intersection_type = split_param[0]
        ambulance_prob = float(split_param[1])
        flow_north_south_prob = float(split_param[2])
        flow_west_east_prob = float(split_param[3])
        seed = int(split_param[4])
        return intersection_type, ambulance_prob, flow_north_south_prob, flow_west_east_prob, seed
    else:
        raise Exception("Unsupported")

if __name__ == "__main__":
    import sys
    env_name = sys.argv[1]
    orig_steps = int(sys.argv[2])
    steps = int(sys.argv[3])
    level = sys.argv[4]
    feature_extractor = sys.argv[5] #"complete"
    ng_margin = None

    exact_model_number = None
    norm_helper = None
    norm_descriptor = None
    norm_violation_filtering = True
    for arg in sys.argv:
        if "--norm" in arg:
            norm_descriptor, norm_helper = get_norm_helper(env_name,arg, level)
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))
        if "--margin" in arg:
            ng_margin = int(arg.replace("--margin", ""))
        if "--no-filter" in arg:
            norm_violation_filtering = False

    if ng_margin is None:
        ng_margin = 50

    level = parse_level(env_name, level)
    setup_and_ext_train(env_name, orig_steps, steps, level, feature_extractor, norm_helper,
                        norm_descriptor = norm_descriptor,
                        exact_model_number=exact_model_number, ng_margin=ng_margin,norm_violation_filtering=norm_violation_filtering)