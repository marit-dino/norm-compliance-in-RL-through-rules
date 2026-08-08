cd ../..
source ./venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 1000_000 originalClassic 1000 dqn complete --exact_mod$1
