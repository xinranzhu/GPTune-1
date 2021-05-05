# set -x 

dataset="susy_10Kn-adult"
ntask=2
bmin=1
bmax=27
eta=3
Nloop=3

# parse results for each run
for expid in S3 S4
do
    python krr_parse_results_history.py -dataset ${dataset} -ntask ${ntask} -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -expid ${expid}
done

# plot for all runs
python krr_plot_history.py -explist S0 S1 S2 S3 S4 -deleted_tuners None -dataset ${dataset} -ntask ${ntask} -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop}


