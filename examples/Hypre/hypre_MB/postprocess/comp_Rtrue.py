import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter
import ast
import numpy as np
import pdb
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amax', type=int, default=2, help='maximum of coeff_a')
    parser.add_argument('--amin', type=int, default=0, help='minimum of coeff_a')
    parser.add_argument('--cmax', type=int, default=2, help='maximum of coeff_c')
    parser.add_argument('--cmin', type=int, default=0, help='minimum of coeff_c')
    parser.add_argument('--ntask', type=int, default=20, help='number of tasks')
    parser.add_argument('--nsample', type=int, default=20, help='number of random samples')
    parser.add_argument('--eta', type=int, default=3, help='eta value in bandit structure')
    parser.add_argument('--bmax', type=int, default=100, help='maximum budget in bandit structure')
    parser.add_argument('--bmin', type=int, default=10, help='minimum budget in bandit structure')
    parser.add_argument('--expid', type=str, default=None, help='experiment id')
    return parser.parse_args()

def main(args):
    my_source = f"./data/Rtruedata_amin{args.amin}_cmin{args.cmin}_amax{args.amax}_cmax{args.cmax}_ntask{args.ntask}_nsample{args.nsample}.txt"
    save_path = f"./data/Rtruedata_amin{args.amin}_cmin{args.cmin}_amax{args.amax}_cmax{args.cmax}_ntask{args.ntask}_nsample{args.nsample}.pkl"

    summary = []
    # summary = [[0, a, c, [PS], [Os]], ... [9, [a, c], [PS], [Os]]]
    i = 0
   
    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split()
            if len(info) > 0 and info[0][1:-1] == "a_val":
                a = float(info[3][1:-1])
                c = float(info[4][:-1])
                if i > 0:
                    # print("i = ", i)
                    # print("data_temp = ", data_temp)
                    summary.append(data_temp)
                data_temp = [i, [a, c]]
                i += 1
                line = f.readline()
            elif len(info) > 0 and info[0] == 'Os':
                info = ast.literal_eval("".join(info[1:]))
                info = np.array(info).reshape((len(info),))
                data_temp.append(info)
                line = f.readline()
            else:
                line = f.readline()
                
    summary.append(data_temp)
    pickle.dump(summary, open(save_path, "wb"))  
    print("summary = ", summary)
    return save_path

def comp_Rtrue(args, datasrc):
    ntask = args.ntask
    results_summary = pickle.load(open(datasrc, "rb"))
    R = np.zeros((ntask, ntask))
    for i in range(ntask):
        Oi = np.concatenate((results_summary[i][2], results_summary[i][3]))
        for j in range(i, ntask):
            Oj = np.concatenate((results_summary[j][2], results_summary[j][3]))
            R[i, j], _ = stats.pearsonr(Oi, Oj)
    
    print("The Rtrue matrix is estimated as \n", R)
if __name__ == "__main__":
    save_path = main(parse_args())
    comp_Rtrue(parse_args(), save_path)