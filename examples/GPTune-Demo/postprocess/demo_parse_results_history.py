import os
import os.path as osp
import argparse
import pickle
import numpy as np
import re
from operator import itemgetter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ntask', type=int, default=1, help='number of tasks')
    parser.add_argument("-Nloop", type=int, default=1, help='number of bandit loops')
    parser.add_argument("-bmin", type=int, default=1, help ='minimum value for bandit budget')
    parser.add_argument("-bmax", type=int, default=8, help ='maximum value for bandit budget')
    parser.add_argument('-expid', type=str, default='0')
    return parser.parse_args()


def main(args):
    summary = []
    my_source = f'./data/demo_ntask{args.ntask}_Nloop{args.Nloop}_expid{args.expid}.txt'
    save_path = f'./data/demo_ntask{args.ntask}_Nloop{args.Nloop}_expid{args.expid}.pkl'
    GPTuneBand_source = f'./data/demo_ntask{args.ntask}_Nloop{args.Nloop}_expid{args.expid}_GPTuneBand_parsed.pkl'
    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split()
            if (info[0] == 'Tuner:' and info[1] == "GPTuneBand"):
                print(line)
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline()
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    tid = int(line[1])
                    line = f.readline().split() 
                    task = float(line[1])
                    line = f.readline()
                    line = f.readline()
                    result = pickle.load(open(GPTuneBand_source, "rb"))
                    results.append(result)
                    if int(args.ntask) > 1:
                        line = f.readline()
                        line = f.readline().split()
                summary.append(results)
                line = f.readline()
                line = f.readline()
                print(results)
            elif (info[0] == 'Tuner:' and info[1] == "hpbandster"):
                print(line)
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline()
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    tid = int(line[1])
                    line = f.readline().split()
                    task = float(line[1])
                    line = f.readline()
                    line = f.readline().strip("    Os  ")
                    data = [[float(y) for y in x.split(", ")] for x in re.split('\[\[|\]\]|\), \(|\(|\)', line) if len(x) > 2]
                    x = []
                    y = []
                    pre_fix = 0
                    max_num = -999
                    for info in data:
                        if info[0] > max_num:
                            max_num = info[0]
                    for info in data:
                        pre_fix += info[0]/max_num
                        if np.isclose(info[0], max_num):
                            x.append(pre_fix)
                            y.append(info[1])
                    results.append([tid, task,  [x, y]]) 
                    if int(args.ntask) > 1:
                        line = f.readline()
                        line = f.readline().split()
                summary.append(results)
                print(results)
                line = f.readline()
                line = f.readline()
            else: # GPTune OpenTuner and TPE
                print(line)
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline()
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    tid = int(line[1])
                    line = f.readline().split() 
                    task = float(line[1])
                    line = f.readline()
                    line = f.readline().strip('    Os  [[ ]\n') 
                    history = []
                    x = []
                    cost = 1.0
                    x.append(cost)
                    history.append(float(line))
                    line = f.readline()
                    while line.startswith(' ['):
                        cost += 1.0
                        x.append(cost)
                        info = line.strip(' [ ]\n')
                        history.append(float(info))
                        line = f.readline()
                    results.append([tid, task, [x,history]])
                    if int(args.ntask) > 1:
                        line = f.readline()
                        line = f.readline().split() 
                summary.append(results)
                print(results)
                line = f.readline()
                
    pickle.dump(summary, open(save_path, "wb"))  

if __name__ == "__main__":
    main(parse_args())

