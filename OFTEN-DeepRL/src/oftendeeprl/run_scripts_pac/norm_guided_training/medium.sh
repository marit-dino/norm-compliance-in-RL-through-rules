cd ../..
source ./venv/bin/activate
python extended_training.py BerkeleyPacman-v0 1000_000 1000_000 mediumClassic complete --norm2-5-True --margin50 --exact_mod$1
