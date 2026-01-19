cd ../..
source .venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 5000_000 originalClassic_no_capsules 1000 dqn --shield117-from_shield-find_comb --exact_mod$1
