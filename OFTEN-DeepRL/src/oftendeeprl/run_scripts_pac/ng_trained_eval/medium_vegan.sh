cd ../..
source ./venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 1000_000 mediumClassic 1000 norm_guided_dqn__vegan_2_5_ complete --ext1000000 --exact_mod$1
