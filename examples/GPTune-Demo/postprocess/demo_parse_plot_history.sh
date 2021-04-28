# set -x 

ntask=1
Nloop=2
expid="4"
fmin=0.7

# read results
python demo_parse_results_history.py  -bmin 1 -bmax 27 -ntask ${ntask} -Nloop ${Nloop} -expid ${expid}
python demo_plot_history.py  -ntask ${ntask} -Nloop ${Nloop} -expid ${expid} -fmin ${fmin}

