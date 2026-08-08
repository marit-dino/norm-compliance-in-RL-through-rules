cd ../..
source .venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 5000_000 originalClassic 250 dqn --shield117-from_shield-random --exact_mod$1
