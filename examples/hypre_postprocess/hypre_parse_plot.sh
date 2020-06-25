<<<<<<< HEAD
set -x 

nmax=200
nmin=10
ntask=60
equation="convdiff"
# equation="Poisson"
=======
nmax=200
nmin=100
ntask=30
equation="convdiff"
>>>>>>> update hypre results
nrun=10

# read results
python parse_tuning_results.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} 
# plot
<<<<<<< HEAD
python hypre_plot.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun}

=======
python hypre_plot.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun}
>>>>>>> update hypre results
