for i in $(seq 0 50);
do python3 /home/christophwelling/RadioNeutrino/pueo/Nu_Flavor/efield_reco/run_full_reco.py $1 el --event_id=$i
done
