# set -x 

nmax=200
nmin=100
ntask=30
equation="convdiff"
# equation="Poisson"
nrun=10
xtype="iter"
# xtype="time"

# read results
python parse_results_MLA_history.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun} 
# plot
python hypre_plot_MLA_history.py  --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun} --xtype ${xtype}
