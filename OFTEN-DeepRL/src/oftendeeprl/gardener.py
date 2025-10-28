import os
import random
import sys
import pickle

from gym_gardener.feature_extractors import MinigridFeaturesExtractorNonDict
from gym_gardener.game import Game
from gym_gardener.learning import Learning
from gym_gardener.instance import create_random_instance, read_from_file

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import torch

from gym_gardener.gym_gardener import GardenerEnv
from util import save_model, save_pickle, load_dqn_model
from gym_gardener.wrappers import create_gardener_env

def default(str):
    return str + ' [Default: %default]'


def readCommand(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python gardener.py <options>
    """
    parser = OptionParser(usageStr)

    parser.add_option('--horizon', dest='horizon', type='int',
                      help=default(
                          'horizon of ASP FR'),
                      metavar='HORIZON', default=5)
    parser.add_option('-t','--steps', dest='steps', type='int',
                      help=default(
                          'number of learning steps'),
                      metavar='STEPS', default=100_000)
    parser.add_option('--radius', dest='radius', type='int',
                      help=default(
                          'radius of ASP FR'),
                      metavar='RADIUS', default=6)
    parser.add_option('--cache', dest='cache', type='int',
                      help=default(
                          'cache of ASP FR'),
                      metavar='RADIUS', default=1)
    parser.add_option('-n', '--numInst', dest='numInst', type='int',
                      help=default(
                          'the number of INSTANCES to generate (per setting)'),
                      metavar='INSTANCES', default=10)
    parser.add_option('-r', '--randSeed', dest='randSeed', type='int',
                      help=default(
                          'the RANDOM SEED for the entire session'),
                      metavar='RANDOM SEED', default=42)
    parser.add_option('-s', '--instSizes', dest='instSizes', help=default(
            'list of instance SIZES to generate (comma separated)'),
                      metavar='INSTANCES', default='10,25,50,100')
    parser.add_option('-p', '--parameter', dest='parameter', help=default(
            'list of parameter for instances to generate (comma separated)'),
                      metavar='parameter', default='0.25,0.05,0.5')
    parser.add_option('--prefix', dest='prefix', help=default(
            'prefix for generated instances'),
                      metavar='parameter', default='small')
    parser.add_option('-m', '--mode', dest='mode', type='int',
                      help=default(
                          '0 - create instances, 1 run rl, 2 run aspfr'),
                      metavar='MODE', default=1)
    parser.add_option('-i', '--interface', dest='interface', action='store_true',
                      help='Show the interface while evaluation',
                      default=False)

    parser.add_option('--instance-seed', dest='instance_seed', type='int',
                      help=default(
                          'Seed for instance generation.'),
                      metavar='SEED', default = None)
    parser.add_option('--view-size', dest='view_size', type='int',
                      help=default(
                          'Size of the square that the agent can see.'),
                      metavar='VIEW SIZE', default = 9)
    parser.add_option('--percentage-walls', dest='pctg_walls', type='float',
                      help=default(
                          'Percentage of the grid with walls [0,1).'),
                      metavar='PERCENTAGE WALLS', default = None)
    parser.add_option('--percentage-frogs', dest='pctg_frogs', type='float',
                      help=default(
                          'Percentage of the grid with frogs [0,1).'),
                      metavar='PERCENTAGE FROGS', default = None)
    parser.add_option('--percentage-phenomena', dest='pctg_phenomena', type='float',
                      help=default(
                          'Percentage of the grid with phenomena [0,1).'),
                      metavar='PERCENTAGE PHENOMENA', default = None)
    parser.add_option('--steps-start-pos', dest='steps_per_start_position', type='int',
                      help=default(
                          'Steps per start position.'),
                      metavar='STEPS START POSITION', default = None)
    
    options, files = parser.parse_args(argv)
    # if (len(files) != 1 and options.mode != 0) or (options.mode == 0 and len(files) != 0):
    #     raise Exception('Command line input not understood: ' + str(files))

    random.seed(options.randSeed)

    args = dict()
    args['numInst'] = options.numInst
    args['instSizes'] = list(map(int, options.instSizes.split(',')))
    args['parameter'] = list(map(float, options.parameter.split(',')))
    args['mode'] = options.mode
    args['interface'] = options.interface
    args['prefix'] = options.prefix
    args['horizon'] = options.horizon
    args['radius'] = options.radius
    args['cache'] = options.cache
    args['steps'] = options.steps
    args['instance_seed'] = options.instance_seed
    args['pctg_walls'] = options.pctg_walls
    args['pctg_frogs'] = options.pctg_frogs
    args['pctg_phenomena'] = options.pctg_phenomena
    args['view_size'] = options.view_size
    args['steps_per_start_position'] = options.steps_per_start_position
    return args, files


def createInstances(numInst, instSizes, parameter, prefix):
    settings = ["deterministic", "nondeterministic"]

    total = len(settings) * len(instSizes) * numInst
    count = 0
    for setting in settings:
        for instSize in instSizes:
            for i in range(numInst):
                count += 1
                instance = create_random_instance(instSize, setting, i,
                                                  parameter[0], parameter[1],
                                                  parameter[2], prefix)
                instance.save()
                printProgressBar(count, total, prefix='Progress:',
                                 suffix='Complete', )


def trainRL():
    total = len(os.listdir('instances'))
    count = 0
    for file in os.listdir('instances'):
        if os.path.isdir('instances/' + file):
            continue

        instance = read_from_file('instances/' + file)
        learning = Learning()
        learning.learn(instance)
        count += 1
        printProgressBar(count, total, prefix='Progress:',
                         suffix='Complete', )


def evaluateRL(files, show):
    instance = read_from_file(files[0])
    learning = pickle.load(
        open('instances/learning/' + instance.name + '.pkl', 'rb'))
    game = Game(instance, learning, show, False)
    game.run()
    instance.print_report('rl')

def evaluateRL_DQN(instSizes, extra_steps, pctg_walls, pctg_phenomena, pctg_frogs, view_size, steps_per_start_position=None, instance_seed=None, show=True):
    
    model = load_dqn_model('pickles/models/dqn_gardener_level_' + str(instSizes[0]) + '_steps_start_positions_' + str(steps_per_start_position) + '_extra_steps_' + str(extra_steps))

    env = create_gardener_env(size=instSizes[-1], pctg_walls=pctg_walls, pctg_phenomena=pctg_phenomena, 
                            pctg_frogs=pctg_frogs, view_size=view_size, instance_seed=instance_seed, )
    game = Game(model, show, False, dqn=True, env=env)
    game.run()

def evaluateFR(files, show, horizon, radius, cache, telingo):
    instance = read_from_file(files[0])
    learning = pickle.load(
        open('instances/learning/' + instance.name + '.pkl', 'rb'))
    game = Game(instance, learning, show, True, horizon=horizon + 1, radius=radius, cache=cache, telingo=telingo)
    game.run()
    instance.print_report('asp')


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                     length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def trainRL_DQN(instSizes, view_size, pctg_walls, pctg_phenomena, pctg_frogs, steps, instance_seed=None, steps_per_start_position = None,
                policy="cnn"):
    tb_name = "dqn"
    tb_path = f"./tb_log/{tb_name}/" if tb_name is not None else None

    algo_name = "dqn"
    env_name = "gardener"
    if policy == "cnn":
        from gym_gardener.feature_extractors import MinigridFeaturesExtractor
        policy_kwargs_dqn = dict(
            features_extractor_class=MinigridFeaturesExtractorNonDict,
            features_extractor_kwargs=dict(features_dim=512,view_size=view_size),
        )
        policy = "MlpPolicy"
    elif policy == "mlp":
        policy_kwargs_dqn = dict(
            net_arch=[256]*4,
            activation_fn=torch.nn.ReLU,
        )
        policy = "MlpPolicy"
    else:
        raise Exception(f"Policy {policy} not implemented. Use cnn or mlp.")

    
    inst_size = instSizes[0]

    # env_aux = create_gardener_env(size=inst_size, pctg_walls=pctg_walls,
    #                                 pctg_phenomena=pctg_phenomena,
    #                                 pctg_frogs=pctg_frogs, view_size=view_size,
    #                                 instance_seed=instance_seed,
    #                                 steps_per_start_position=steps_per_start_position)
    #
    # print(f"States reaching the goal: {len(env_aux.unwrapped.positions_reaching_goal)}")
    # steps = len(env_aux.unwrapped.positions_reaching_goal)*steps_per_start_position
    # print(f"Training steps: {steps}")

    env = make_vec_env(lambda : create_gardener_env(size=inst_size, pctg_walls=pctg_walls,
                                                    pctg_phenomena=pctg_phenomena,
                                                    pctg_frogs=pctg_frogs, view_size=view_size,
                                                    instance_seed=instance_seed,
                                                    steps_per_start_position=steps_per_start_position),
                                                    n_envs=16)

    model = DQN(policy, env, device="cuda", verbose =1, tensorboard_log=tb_path, policy_kwargs=policy_kwargs_dqn,
                batch_size=256, buffer_size=50_000, exploration_fraction=0.5, gamma = 0.95,gradient_steps=-1
                )
    
    # Stop training if there is no improvement after more than 10 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=20, verbose=1)
    eval_callback = EvalCallback(env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    # The training will stop as soon as the the number of consecutive evaluations without model
    # improvement is greater than 10
    model.learn(steps, callback=eval_callback)

    # print(f"Training steps: {steps}")
    #
    # env = make_vec_env(lambda : create_gardener_env(size=inst_size, pctg_walls=pctg_walls,
    #                                                 pctg_phenomena=pctg_phenomena,
    #                                                 pctg_frogs=pctg_frogs, view_size=view_size,
    #                                                 instance_seed=instance_seed),
    #                                                 n_envs=16)
    #
    # model.set_env(env)
    # model.learn(steps)
    # model_name = f"pickles/models/{algo_name}_{env_name}_level_{instSizes[0]}_steps_start_positions_{steps_per_start_position}_extra_steps_{extra_steps}"
    level_add = f"{pctg_walls}_{pctg_phenomena}_{pctg_frogs}".replace(".","-")
    model_name = f"pickles/models/{algo_name}_{env_name}_{steps}_level_{instSizes[0]}_{level_add}"

    save_model(model_name,model)
    if instance_seed is not None:
        instance = env.get_attr("instance")
        save_pickle(model_name.replace("models","gardener_instances"),instance)

if __name__ == '__main__':
    """
    The main function called when gardener.py is run
    from the command line:

    > python gardener.py

    See the usage string for more details.

    > python gardener.py --help
    """
    args, files = readCommand(sys.argv[1:])

    if args['mode'] == 0:
        # createInstances(args['numInst'], args['instSizes'], args['parameter'], args['prefix'])
        # trainRL()
        trainRL_DQN(instSizes=args['instSizes'], steps=args['steps'], pctg_frogs=args['pctg_frogs'],
                    pctg_walls=args['pctg_walls'], pctg_phenomena=args['pctg_phenomena'], 
                    instance_seed=args['instance_seed'], view_size=args['view_size'], 
                    steps_per_start_position=args['steps_per_start_position'])
    if args['mode'] == 1:
        # evaluateRL(files, args['interface'])
        evaluateRL_DQN(instSizes=args['instSizes'], extra_steps=args['steps'], pctg_walls=args['pctg_walls'],
                       pctg_phenomena=args['pctg_phenomena'], pctg_frogs=args['pctg_frogs'], view_size=args['view_size'],
                        instance_seed=args['instance_seed'], show=True, 
                        steps_per_start_position=args['steps_per_start_position'])
    if args['mode'] == 2:
        evaluateFR(files, args['interface'], args['horizon'], args['radius'], args['cache'], False)
    if args['mode'] == 3:
        evaluateFR(files, args['interface'], args['horizon'], args['radius'], args['cache'], True)


# pyinstaller --onefile gardener.py --hidden-import=_cffi_backend
# gardener -a instance.txt learning.txt