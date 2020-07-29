# set -x 

nmax=200
nmin=100
ntask=30
equation="convdiff"
# equation="Poisson"
# nrun=10
# xtype="iter"
# xtype="time"

taskid=0 # single plot of a given task, sorted
multistart=0 # 0 or 5
ratio2best="anytime" # anytime, alltime, or None
# ratio2best="alltime"
# ratio2best="None"
for nrun in 10
do
    # read results
    # python parse_results_MLA_history.py --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun} --multistart ${multistart}
    # plot
    xtype="iter"
    python hypre_plot_MLA_history.py  --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun} --xtype ${xtype} --multistart ${multistart} --ratio2best ${ratio2best} --taskid ${taskid} 2>&1 | tee out.historical_mean_set_${equation}_nmax${nmax}_nmin${nmin}_nrun${nrun}
    # xtype="time"
    # python hypre_plot_MLA_history.py  --nmax ${nmax} --nmin ${nmin} --ntask ${ntask} --equation ${equation} --nrun ${nrun} --xtype ${xtype} --multistart ${multistart} --ratio2best ${ratio2best} --taskid ${taskid}
done
