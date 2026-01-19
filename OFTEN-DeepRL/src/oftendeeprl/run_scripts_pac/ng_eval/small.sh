cd ../..
source ./venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 1000_000 smallClassic 1000 dqn complete --norm2-5-True --exact_mod$1
