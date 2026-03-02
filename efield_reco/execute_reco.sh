for i in $(seq 0 20);
do python3 run_full_reco.py $1 --event_id=$i
done