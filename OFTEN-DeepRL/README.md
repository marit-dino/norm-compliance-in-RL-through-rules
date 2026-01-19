The code from this repository implements the OFTEN-DeepRL methodology and it includes all the necessary scripts to evaluate its performance. The description of the methodology can be found in the paper presented at ECAI 2025 titled "_OFTEN-DeepRL: On-the-Fly Teaching of Ethical Norms to Deep Reinforcement Learning Agents_".

OFTEN-DeepRL is an approach to integrate ethical norms into agents trained with deep reinforcement learning. It starts by training an RL policy focused on task performance. Building upon such a pre-trained policy, OFTEN-DeepRL adapts the policy through norm-guided training. For a combination of observations and domain knowledge, we employ a logic program that generates norm-compliant plans for the agent using answer set programming (ASP) within a given planning horizon. These plans serve as demonstrations for fine-tuning the agent's policy in the norm-guided training phase, guiding it toward behavior that remains effective while respecting the specified norms.

Our approach has been validated with three types of scenarios: Pac-Man, a gardener simulation, and a SUMO-RL traffic control scenario. In all settings, agents fine-tuned with OFTEN-DeepRL achieve comparable task performance while significantly reducing norm violations.

# Citation

To cite this work in your research, please use the following BibTeX entry:

```
@inproceedings{lopez:ECAI25,
  author       = {Ignacio D. Lopez{-}Miguel and
                  Sebastian Adam and
                  Ezio Bartocci and
                  Thomas Eiter and
                  Martin Tappler},
  title        = {{OFTEN-DeepRL}: On-the-Fly Teaching of Ethical Norms to Deep Reinforcement Learning Agents},
  booktitle    = {{ECAI} 2022 - 28th European Conference on Artificial Intelligence,
                  October 25-30, 2025, Bologna, Italy},
  publisher    = {{IOS} Press},
  year         = {2025},
}
```

# Organization

The high-level structure of the most important files is described below:

- [gym_pacman](gym_pacman), [gym_sumo](gym_sumo), and [gym_gardener](gym_gardener) contain the utilities to wrap the differen scenarios as gymnasium environments.
- [sb3_ext](sb3_ext) includes the code to perform RL. It is based on stable-baselines3 but extended as described in the methodology for OFTEN-DeepRL.
- *.lp files are the logic programs.
- [run_scripts_pac](run_scripts_pac), [run_scripts_gardener](run_scripts_gardener), and [run_scripts_sumo](run_scripts_sumo) include the scripts that allow one to run all the experiments. The scripts from these folders make use of the following files:
    - [train_pacman.py](train_pacman.py) to train Pac-Man.
    - [gardener.py](gardener.py) to train Gardener.
    - [sumo_single_intersection.py](sumo_single_intersection.py) to train SUMO.
    - [evaluate_policy.py](evaluate_policy.py) to evaluate the trained policies.
    - [extended_training.py](extended_training.py) to extend the Base policy with the norm-guided training. It is also used for the ablation studies.

# Installation

The code has been tested in Ubuntu-24.04 with Python 3.12.3.

The set of packages that were used to run the experiments are found in the requirements.txt file. Some of the packages listed in this file might not be necessary, but it was decided to let them to be sure of the replicability of the results.  It is recommended to create a new virtual environment and, then, install the packages:

```
cd ~/OFTEN-DeepRL # go to the root of the folder containing the code for OFTEN-DeepRL
python -m venv ./venv
python -m pip install -r requirements.txt
```

For SUMO-RL, one also needs to install SUMO:

```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

One needs to include the `SUMO_HOME` environment variable. This is done by adding the following in the `.bashrc` file if `/usr/share/sumo` is the folder in which SUMO has been installed. Otherwise, it has to be replaced by the right directory.

```
export SUMO_HOME="/usr/share/sumo"
```

Furthermore, in order to accelerate the simulations for SUMO, one needs to include the `LIBSUMO_AS_TRACI` environment variable. This is done by adding the following in the `.bashrc` file.

```
export LIBSUMO_AS_TRACI=1
```

For Gardener, it is necessary to install the package `tkinter`.

```
sudo apt-get install python3-tk
```

# Norm-Guided-RL

There are three folders containing the scripts to launch the experiments for the three scenarios:
- [run_scripts_pac](run_scripts_pac) for Pac-Man,
- [run_scripts_gardener](run_scripts_gardener) for Gardener, and
- [run_scripts_sumo](run_scripts_sumo) for SUMO.

The three of them are organized in the following subfolders:
- **training**: it trains the Base policy.
- **evaluate**: it evaluates the trained Base policy obtained from the script in training.
- **ng_eval**: it evaluates the trained Base policy using Policy Fixes.
- **norm_guided_training**: it extends the training from the Base policy using the norm guided training OFTEN-DeepRL methodology.
- **ng_trained_eval**: it evaluates the trained policy obtained from the **norm_guided_training_*** folders.

Since further ablation studies were performed for Gardener and SUMO, the scripts related to those can be found in the following folders:
- **norm_guided_training_abl_no_exp**: it extends the training from the Base policy using the norm guided training OFTEN-DeepRL methodology with a margin of 0 (no expert).
- **norm_guided_training_abl_no_filter**: it extends the training from the Base policy using the norm guided training OFTEN-DeepRL methodology without filtering the norm-violating experiences.
- **norm_guided_training_fresh**: it uses the norm guided training OFTEN-DeepRL methodology from scratch, that is, without taking a Base policy and extending it. 
- **ng_trained_abl**: it evaluates the different ablation studies.

Each of the previous folder contains the scripts for all the different settings for each scenario. As an example, we will show how the scripts can be used to train and evaluate Gardener in the 25x25 grid with the setting in which there are frogs and plants. _If one wants to see faster but worse results, one can change the number of steps from 250_000 to a lower number in the following scripts, as well as the number of evaluations from 1000 to a lower number._ The models and data produced in the experiments is not included in the .zip file due to size limitations but they can be obtained by running the scripts.

1. Training the base policy:
```
cd OFTEN-DeepRL/run_scripts_gardener/training
bash 25x25_frogsandplants.sh 
```
It will save the model in pickles/models named `dqn_gardener_250000_level_25_0-25_0-05_0-05_1.zip`.

2. Evaluating the base policy:
```
cd OFTEN-DeepRL/run_scripts_gardener/evaluate
bash 25x25_frogsandplants.sh 
```

It will print the results and save the data in pickles/eval_stats/base in the file `dqn_gardener_250000_level_25_0-25_0-05_0-05_1.pkl`.

3. Extending the training using OFTEN-DeepRL methodology:
```
cd OFTEN-DeepRL/run_scripts_gardener/norm_guided_training
bash 25x25_frogsandplants.sh 
```
It will save the model in pickles/models named `norm_guided_dqn___2_4__gardener_250000_level_25_0-25_0-05_0-05_1.zip`.

4. Evaluating the trained policy using OFTEN-DeepRL:
```
cd OFTEN-DeepRL/run_scripts_gardener/ng_trained_eval
bash 25x25_frogsandplants.sh 
```
It will print the results and save the data in pickles/eval_stats/extended in the file `norm_guided_dqn___2_4__gardener_250000_to_125000_level_25_0-25_0-05_0-05_1.pkl`.

Due to the stochastic nature of the environment, it is possible that the results differ from the ones reported on the paper. The ones reported on the paper result from an average of 5 runs with different seeds. The scripts for Gardener and SUMO have been fixed to the seed 1337. This can be change directly in the scripts.

To run the ablation studies for SUMO and Pac-Man, one can follow a similar proccedure as shown for the general training and evaluation part.

## Logic Programs

All logic programs follow the policy fixing approach described in [ASP-Driven Emergency Planning for Norm Violations in Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/33619).
In the following the exact penalties and rewards defined in the logic programs are detailed.

### [Gardener](gardener_program.lp)
The rewards from following policy preferences and penalties from violating a norm are encoded in the logic program.
In our implementation, all rewards and penalties are accounted on the same priority level.
Penalties for killing a frog or plant are equal (5) and higher than any potential reward of a single action (0-3).
To discourage the agent from looping in a norm-compliant trajectory, an additional penalty (10), sanctioning plans containing repeated states, is introduced.
Similarly, an additional reward (5) when reaching the target is established and rewards received from following policy preferences are multiplied by an increasing number when the agent re-visits a cell.
Earlier violations and norm adherence are weighted higher to discourage immediate violations or policy deviations. 

### [Pac-Man](pacman_program.lp)
To account for the different norms (vegan and vegetarian), [pacman_helper.py](sb3_ext/pacman_helper.py) excludes setting the external atoms containing information about the blue ghost in the vegetarian case.
Therefore, the logic program is “blind” regarding this ghost and will not hinder Pacman from eating it.

The rewards from following policy preferences and penalties from violating a norm are encoded in the logic program.
In our implementation, norm violations are accounted at a higher priority level (2), than rewards (1).
The reward of an action is based on policy preference (0-3), where 0 is the preferred action by the policy and a higher number indicates a less preferred action.
Earlier violations are weighted higher to discourage immediate violations.

### [SUMO](gym_sumo/program.lp)
The static information about an instance is automatically generated and added to the logic program by [sumo_helper.py](sb3_ext/sumo_helper.py).
The delta between time steps is currently harcoded to 5 reflected by Line 14 in [program.lp](gym_sumo/program.lp), as our SUMO implementation uses a time step size of 2 and we account for an average start-up delay of 3 before vehicles start moving again once the traffic light turns green.

The rewards from following policy preferences and penalties from violating a norm are encoded in the logic program.
In our implementation, all rewards and penalties are accounted on the same priority level.
Since there are only two potential actions at any point in time the reward for following an action is hardcoded to 10.
The penalty for violating the norm is equal to the time waited W by the ambulance in seconds.