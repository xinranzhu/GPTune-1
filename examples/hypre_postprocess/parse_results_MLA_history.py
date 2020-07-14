import os
import os.path as osp
import argparse
import pickle
import numpy as np
from operator import itemgetter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmax', type=int, default=100, help='maximum discretization size')
    parser.add_argument('--nmin', type=int, default=10, help='minimum discretization size')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument("--equation", type=str, default="Poisson", help ='type of PDE to solve')
    parser.add_argument("--nrun", type=str, default="10", help ='number of hypre runs')
    return parser.parse_args()


def main(args):
    summary = []
    my_source = f'./data_MLA_history/exp_hypre_history_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_nrun{args.nrun}.txt'
    save_path = f'./data_MLA_history/exp_hypre_history_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_nrun{args.nrun}.pkl'
    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split()
            if (info[0] == 'Tuner:' and info[1] != "HpBandster"):
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    try:
                        tid = int(line[1])
                    except:
                        print(line)
                        print(results)
                    line = f.readline().split() 
                    nx = int(line[0][3:])
                    ny = int(line[1][3:])
                    nz = int(line[2][3:])
                    line = f.readline()
                    line = f.readline().split('[[') 
                    try:
                        history = [float(line[1].strip(' ]\n'))]
                    except:
                        print(f'ERROR, {tid}')
                        print(line)
                        print()
                        print(line[1].strip(' ]'))
                        print()
                    # for _ in range(int(args.nrun) - 1):
                    line = f.readline()
                    while line.startswith(' ['):
                        info = line.strip(' [ ]\n')
                        try:
                            history.append(float(info))
                        except:
                            print(line)
                        line = f.readline()
                    results.append([tid, (nx, ny, nz), history]) 
                    # line = f.readline().split()
                    line = f.readline().split()
                results[1:] = sorted(results[1:], key=lambda x:np.prod(x[1]))
                summary.append(results)
                line = f.readline()
            else:
                results = []
                tunername = info[1]
                results.append(tunername)
                line = f.readline().split() 
                for _ in range(int(args.ntask)):
                    tid = int(line[1])
                    line = f.readline().split()
                    nx = int(line[0][3:])
                    ny = int(line[1][3:])
                    nz = int(line[2][3:])
                    line = f.readline()
                    line = f.readline().strip('    Os').split()
                    history = [float(line[0][2:-2])]
                    for i in range(int(args.nrun)-1):
                        history.append(float(line[i+1][1:-2]))
                    results.append([tid, (nx, ny, nz), history]) 
                    line = f.readline()
                    line = f.readline().split()
                results[1:] = sorted(results[1:], key=lambda x:np.prod(x[1]))
                summary.append(results)

    print(summary[0])
    print()
    print(summary[1])
    print()
    print(summary[2])
    print()
    pickle.dump(summary, open(save_path, "wb"))  

if __name__ == "__main__":
    main(parse_args())

