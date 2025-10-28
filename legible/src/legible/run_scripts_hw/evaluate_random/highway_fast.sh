cd ../..
source .venv/bin/activate
python evaluate_policy.py highway-fast-v0 500_000 0 100 dqn --rand$1-100 --exact_mod$2
