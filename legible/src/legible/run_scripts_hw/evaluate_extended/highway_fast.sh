cd ../..
source .venv/bin/activate
python evaluate_policy.py highway-fast-v0 500_000 0 200 ext_dqn --ext1000000 --exact_mod$1
