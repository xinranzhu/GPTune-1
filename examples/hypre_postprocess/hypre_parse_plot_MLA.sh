# set -x 

nmax=150
nmin=30
ntask=30
# equation="convdiff"
equation="Poisson"

# multistart=None
tuner4=0
rerun=0
average=0
# bandit=1

for nrun in 10 20 30
do
    # read results
    python parse_results_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --tuner4 ${tuner4} --rerun ${rerun} 
    # plot
    python hypre_plot_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun} --tuner4 ${tuner4} --rerun ${rerun} --average ${average}
done

# parse results from unfinished experiments
# nodes=4
# equation='covdiff'
# tuner='GPTune'
# python parse_unfinished.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --nodes ${nodes} --nrun ${nrun} --tuner ${tuner} --equation ${equation} --multistart ${multistart}