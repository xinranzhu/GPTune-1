# set -x 

nmax=100
nmin=10
ntask=30
# equation="convdiff"
equation="Poisson"
nrun=20
# read results
python parse_results_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} 
# plot
python hypre_plot_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun}

