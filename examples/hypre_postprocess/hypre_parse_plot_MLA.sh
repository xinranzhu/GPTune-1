# set -x 

nmax=100
nmin=10
ntask=30
# equation="convdiff"
equation="Poisson"

multistart=5
tuner4=0
rerun=0
average=0
# bandit=1

for nrun in 20
do
    # read results
    # python parse_results_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --tuner4 ${tuner4} --rerun ${rerun} --multistart ${multistart}
    # plot
    python hypre_plot_MLA.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun} --tuner4 ${tuner4} --rerun ${rerun} --average ${average} --multistart ${multistart}
done

# parse results from unfinished experiments
# nodes=4
# equation='covdiff'
# tuner='GPTune'
# python parse_unfinished.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --nodes ${nodes} --nrun ${nrun} --tuner ${tuner} --equation ${equation} --multistart ${multistart}