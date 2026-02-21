# Ensuring Norm Compliance in Reinforcement Learning Agents through Rule Revision

This repository contains the code for my master's thesis ''Ensuring Norm Compliance in Reinforcement Learning Agents through Rule Revision''.

The project aims to improve the norm adherence of reinforcement learning agents using symbolic rules. Initially, an agent is trained on certain norms using the method OFTEN-DeepRL [[1]](#1).
In the second step, symbolic rules that describe the agent's behavior are mined using LEGIBLE [[2]](#2). Both of these methods have been extended to accommodate the interplay between them as well as additional norms that have not been considered in [[1]](#1).
Then the mined rule set is updated by removing and adding rules to it, with the goal of reducing the number of violations when the agent is guided by them. To check whether violations can be reduced, we use ASP planning.
The ASP program is based on this [ASP program](https://gitlab.tuwien.ac.at/martin.tappler/OFTEN-DeepRL/-/blob/3546dfd846f98e7f542b6e6107454bd5e2a5539c/pacman_program.lp) from OFTEN-DeepRL.

To evaluate the approach, several norms have been tested in the Pacman environment:
- **vegan**: Pacman is not allowed to eat ghosts.
- **vegetarian**: Pacman is only allowed to eat one specific ghost.
- **permissive**: Pacman starts out as vegan, but after the first violation he is allowed to eat ghosts.
- **ctd** (contrary-to-duty): Pacman is vegan, but if he eats a ghost, he should eat it when moving north.

## Organization
The project is organized as follows:
- [OFTEN-DeepRL](OFTEN-DeepRL): contains the (adapted and extended) code of [[1]](#1).
- [legible](legible): contains the (adapted) code of [[2]](#2).
- [main](main): contains the code of this project, where:
  - [features.txt](main/features.txt) explains the different features used by the feature extractor and the symbolic rules.
  - [conf](main/conf) contains several config files specifying e.g. the number of evaluation or training episodes, see [here](#config) for more details
  - [src](main/src/) contains the following:
    - [train_policy.py](main/src/train_policy.py) trains the initial norm guided policy, whose rules will be mined
    - [mine_rules.py](main/src/mine_rules.py) mines the rules
    - [check_norms.py](main/src/check_norms.py) provides methods that check whether the current state violates a norm
    - [update_rules.py](main/src/update_rules.py) updates the rule set to reduce norm violations
    - [evaluate_updates.py](main/src/evaluate_updates.py) tests both the policy guided by the original (mined) rule set and also by the updated rule set
    - [util.py](main/src/util.py) provides methods used for setup
    - [rule_util.py](main/src/rule_util.py) provides methods used for interacting with the rule set
    - [pacman_helper_asp.py](main/src/pacman_helper_asp.py) builds upon the PacmanClingoHelper from OFTEN-DeepRL to accommodate checking whether violations can be reduced as well as additional norms
    - [pacman_program_asp.lp](main/src/pacman_program_asp.lp) builds upon [pacman_program.lp](https://gitlab.tuwien.ac.at/martin.tappler/OFTEN-DeepRL/-/blob/3546dfd846f98e7f542b6e6107454bd5e2a5539c/pacman_program.lp) from OFTEN-DeepRL to accommodate checking whether violations can be reduced as well as additional norms



## Installation
After downloading or cloning the repository, follow these steps:

1. If not already installed, install [poetry](https://python-poetry.org/) 
2. Navigate to [main/src](main/src/) and run `poetry install`.

The code has been tested on Ubuntu 24.04 and poetry 2.1.2.


## Running the Code
The project consists of several steps which should be executed in the following order
(Note that all of these commands are executed in [main/src](main/src/)):

1. Train the initial norm guided policy by running `poetry run python train_policy.py`. \
By default, the agent is trained on the norm **vegetarian**. To select other norms, run e.g. `poetry run python train_policy.py norm=vegan` or change the default in [config.yaml](main/conf/config.yaml). The other available norms are **permissive** and **ctd**. Other parameters can be set in a similar fashion, see [here](#config) for more details. After completion, the policy is stored in [main/src/pickles/models](main/src/pickles/models/).

2. Mine rules by running `poetry run python mine_rules.py`. \
As before you can specify other norms (or parameters) by adding e.g. `norm=vegan` to the command.
The mined rules are stored in [main/src/pickles/shields](main/src/pickles/shields/).

3. Update the rule set by running `poetry run python update_rules.py`. \
As before you can specify other norms (or parameters) by adding e.g. `norm=vegan` to the command.
The updated rule set is stored in [main/src/pickles/shields](main/src/pickles/shields/).

4. Evaluate the updates by running `poetry run python evaluate_updates.py`. \
As before you can specify other norms (or parameters) by adding e.g. `norm=vegan` to the command.
The results are stored in [main/src/pickles/eval_stats](main/src/pickles/eval_stats/).


### Config: 
The config is handled by [hydra](https://hydra.cc), and all config files are located in [main/conf](main/conf/). Some norm specific aspects are located in [main/conf/norm](main/conf/norm), however it should not be necessary to change them. The rest of the config parameters (in [config.yaml](main/conf/config.yaml)) can either be changed directly in the file, or by overriding it when executing a command. 
This can be done by specifying the desired value like this:
```
poetry run python update_rules.py norm=permissive
poetry run python update_rules.py rules.updates.episodes=500
poetry run python update_rules.py asp.horizon=2 asp.radius=4
```

## References
<a id="1">[1]</a> 
Ignacio D. Lopez-Miguel, Sebastian Adam, Ezio Bartocci, Thomas Eiter, and Martin Tappler. 
OFTEN-DeepRL: On-the-fly teaching of ethical norms to deep reinforcement learning agents. 
In ECAI 2025 - 28th European Conference on Artificial Intelligence, October 25-30,
2025, Bologna, Italy. IOS Press, 2025.

<a id="2">[2]</a>
Martin Tappler, Ignacio D. Lopez-Miguel, Sebastian Tschiatschek, and Ezio Bartocci. 
Rule-Guided Reinforcement Learning Policy Evaluation and Improvement. 
In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI 2025, Montreal, Canada, August 16-22, 2025, pages 6254–6262. ijcai.org, 2025.



