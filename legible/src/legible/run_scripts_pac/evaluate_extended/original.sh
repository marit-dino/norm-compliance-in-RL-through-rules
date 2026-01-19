cd ../..
source .venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 5000_000 originalClassic 2500 ext_dqn --ext5000000 --exact_mod$1
