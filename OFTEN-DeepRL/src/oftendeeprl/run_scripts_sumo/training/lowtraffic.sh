cd ../..
source ./venv/bin/activate
python sumo_single_intersection.py -t 50_000 -intersection_type singleIntersection -ambulance_prob 0.01 -flow_north_south_prob 0.05 -flow_west_east_prob 0.05 -seed 1337