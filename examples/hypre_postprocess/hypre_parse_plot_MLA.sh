# set -x 

nmax=200
nmin=10
ntask=60
equation="convdiff"
# equation="Poisson"
nrun=10

# read results
python parse_results_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} 
# plot
python hypre_plot_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun}

