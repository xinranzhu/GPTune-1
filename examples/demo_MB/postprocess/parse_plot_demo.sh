ntask=10
bmin=1
bmax=27
eta=3
expid='R25-R29'
width=0.05

python ./parse_results_demo.py --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid}

# python ./plots_demo.py --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid} --width ${width}