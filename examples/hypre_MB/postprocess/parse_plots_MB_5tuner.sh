
amin=0
cmin=0
amax=1
cmax=1
ntask=10
bmin=1
bmax=8
eta=2
expid='10-14'
width=0.05

python ./parse_results_MB_5tuner.py --amax ${amax} --amin ${amin} --cmax ${cmax} --cmin ${cmin} --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid}

python ./plots_MB_5tuner.py --amax ${amax} --amin ${amin} --cmax ${cmax} --cmin ${cmin} --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid} --width ${width}