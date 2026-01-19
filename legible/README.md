# LEGIBLE

This is the implementation of LEGIBLE, which is short for poLicy Evaluation GuIded By ruLEs, on
case studies involving two types of RL environments: PAC-man levels and different environments available
in highway-env package.
It accompanies the paper "Rule-Guided Reinforcement Learning Policy Evaluation and Improvement" by
Martin Tappler, Ignacio D. Lopez-Miguel, Sebastian Tschiatschek, and Ezio Bartocci.
In short, LEGIBLE is a rule-based framework to create explanations of an RL policy's decisions and 
to evaluate and improve the RL policy under consideration, supported by domain knowledge. 
The domain knowledge can be formalized using metamorphic relations that how actions of an RL agent should 
change in response to a change of the state. For more information, we refer to the paper. Below, 
we will rather discuss:
* Dependencies and setup
* How to run the experiments performed for the evaluation presented in the paper. 
* The content of this source code package and how it relates to the specific part of LEGIBLE, 
assuming knowledge about the paper's contents

## Dependencies and Setup
The code is implemented in Python 3.8+ with dependencies listed in a `requirements.txt` file. 
In addition to external dependencies, this code package includes an adapted version of the 
`wittgenstein` libraries for learning rules by Ilan Moscovitz, released under an MIT license 
(see also `wittgenstein/LICENSE.txt`). Additionally, it includes an adapted version of a OpenAI Gym 
wrapper for the [Berkeley Pacman AI engine](http://ai.berkeley.edu/project_overview.html), 
publicly hosted at [GitHub](https://github.com/sohamghosh121/PacmanGym).

### Setup
We recommend to use a virtual environment to install the required dependencies and work with the code. 
For this purpose, navigate to the directory, where you extract the source code, i.e., the location 
of this README.md. Next, execute the following commands.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gymnasium==0.29.1
```
The last line is required due to dependency version conflicts. Upon installation, `highway-env` requires 
`gymnasium>=1.0.0a2`. However, other parts of the project require `gymnasium < 1.0.0`. Since `highway-env` 
still works with `gymnasium v0.29.1`, we apply this workaround (it produces an error message, though). 
Note that we suggest `.venv` as name 
of the virtual environment as this is the default of the popular PyCharm IDE. 

## Running Experiments and Content
The content of the implementation and the files for running the experiments are structured similarly 
to our paper presented LEGIBLE. Hence, the project is best understood with this structure in mind, which 
is reflected in bash scripts available in subdirectories of `run_script_pac`, for running PAC-man, 
and `run_script_hw`, for running experiments with the `highway-env` environments. 
Experimental results presented in the paper can be found in the directory `reference_pickles`. 
Upon running experiments, results will be stored in subdirectories of `pickles`.

Next, we are going to discuss the types of experiments using the names of script directories.
* `training` : Training of an RL agent in a particular environment, a step required before LEGIBLE can be used. 

All scripts that produce result files add an identifier to the end of the file name, starting from 
the number `1`. This helps create a mechanism to match artifacts produced along the way, like matching 
rules to the RL policy from which it was mined. For this mechanism, all Python scripts that load 
artifact, like an RL, take a command line `--exact_mod<n>`, where `n` is a natural number specifying 
the identifier of the artifact to be loaded. This identifier will also be appended during saving. 
All bash scripts discussed below simply take a natural number as argument and passed it to the Python
scripts with `--exact_mod<n>`.

* `create_rules`: Mine rules from an RL agent, i.e., the first actual step of LEGIBLE.
* `generalize`: Generalize rules to other situation. The metamorphic relations are efficiently implemented in 
the Python script `create_rules_pacman.py` and `create_rules_hw.py`.
* `evaluate`: Run the RL policy without enforcing and while each set of rules separately. 
This builds the basis for identifying weaknesses in Section 7.1
* `random_rules`: Creates randomized rules that are needed for `èvaluate_random_rules`. 
Those two sets of scripts form the basis for the RR baseline.
* `evaluation_random`: Evaluates the RL policy while randomly blocking to implement the 
second baseline, called RT.
* `extended_training`: Those scripts load an RL and perform additional training.
* `evaluate_extended`: Evaluation of an RL policy after extended training without enforcing. This
is the baseline reported in Section 7.2 on policy improvements
* `evaluate_combination`: Evaluate an RL policy while enforcing a growing set of, that is, 
it is an implementation of Algorithm 2 from the paper. 

It is easiest to explore the source code starting from these points.
Additionally, there is a `stats_calc` directory, which include scripts to produce 
the tables shown in the paper. 

### Note
If models found in `reference_pickles` cannot be loaded, there is likely a version mismatch 
between packages with which the models were saved and the used packages.
It should be possible to load all result pickle files, as those are plain Python dictionaries.
Maybe it is necessary to import the class `EvalStats` from `evaluate_policy.py`
