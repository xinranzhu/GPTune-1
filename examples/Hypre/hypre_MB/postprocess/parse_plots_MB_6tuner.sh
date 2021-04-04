
amin=0
cmin=0
amax=1
cmax=1
bmin=1
bmax=8
eta=2
ntask=10
expid='0-4-comb64'
width=0.05

python ./parse_results_MB_6tuner.py --amax ${amax} --amin ${amin} --cmax ${cmax} --cmin ${cmin} --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid}

python ./plots_MB_6tuner.py --amax ${amax} --amin ${amin} --cmax ${cmax} --cmin ${cmin} --ntask ${ntask} --bmin ${bmin} --bmax ${bmax} --eta ${eta} --expid ${expid} --width ${width}