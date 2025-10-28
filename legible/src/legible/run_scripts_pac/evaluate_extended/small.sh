cd ../..
source .venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 2500_000 smallClassic 2500 ext_dqn --ext2500000 --exact_mod$1
