

bmin=1
bmax=8
eta=2
ntask='10'
expid='20-24'
width=0.05

python ./parse_results_corr_3tuner.py --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid}

tuner=3
python ./plots_corr_3tuner.py --tuner ${tuner} --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid}
