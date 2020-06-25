nmax=150
nmin=50
ntask=30
nrun=10

txt_file = "./exp_hypre_summary_nmax${nmax}_nmin${nmin}_ntask{$ntask}.txt"
pkl_file = "./exp_hypre_summary_nmax${nmax}_nmin${nmin}_ntask{$ntask}.pkl"

# read results
python parse_tuning_results.py --source txt_file --save_path pkl_file
# plot
python hypre_plot.py --source pkl_file --nrun ${nrun}