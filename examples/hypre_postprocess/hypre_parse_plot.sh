nmax=200
nmin=100
ntask=30
equation="convdiff"
nrun=10

# read results
python parse_tuning_results.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} 
# plot
python hypre_plot.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun}