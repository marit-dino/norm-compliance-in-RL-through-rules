cd ../..
source ./venv/bin/activate
python evaluate_policy.py BerkeleyPacman-v0 1000_000 smallClassic 1000 norm_guided_dqn_no_filter__vegan_2_5_ complete --ext1000000 --exact_mod$1
