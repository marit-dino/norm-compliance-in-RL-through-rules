cd ../..
source .venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 5000_000 originalClassic 250 dqn --rand$1-100 --exact_mod$2
