# set -x 

nmax=100
nmin=10
ntask=30
# equation="convdiff"
equation="Poisson"
nrun=30
# multistart=5
bandit=1
# read results
python parse_results_TPE_bandit.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation}
# plot
python hypre_plot_TPE_bandit.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun}

