# set -x 

ntask=30
# equation="convdiff"
equation="Poisson"
nrun=20
# typeofresults="performance"

# read results
python parse_results_singletask.py --ntask ${ntask} --equation ${equation} --typeofresults "performance"
python parse_results_singletask.py --ntask ${ntask} --equation ${equation} --typeofresults "time"
# plot
python hypre_plot_singletask.py --ntask ${ntask} --equation ${equation} --nrun ${nrun} 
