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
    parser.add_argument("--equation", type=str, default="Poisson", help ='ML or covdiff')
    parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--nrun', type=int, default=10, help='number of hypre runs')
    parser.add_argument("--tuner", type=str, default="GPTune", help ='GPTune or OpenTuner or HpBandster')
    parser.add_argument('--multistart', type=int, default=None, help='number of model restarts')
    parser.add_argument('--bandit', type=int, default=None, help='use bandit strategy in HpBandster or not')
    return parser.parse_args()


def main(args):
    if args.multistart == None and args.bandit == None:
        my_source = f"./a.out_hypre_{args.equation}_nxmax{args.nmax}_nymax{args.nmax}_nzmax{args.nmax}_nxmin{args.nmin}_nymin{args.nmin}_nzmin{args.nmin}_nodes{args.nodes}_core32_ntask{args.ntask}_nrun{args.nrun}_{args.tuner}"
        save_path = f"./out_hypre_{args.equation}_nxmax{args.nmax}_nymax{args.nmax}_nzmax{args.nmax}_nxmin{args.nmin}_nymin{args.nmin}_nzmin{args.nmin}_nodes{args.nodes}_core32_ntask{args.ntask}_nrun{args.nrun}_{args.tuner}.pkl"
    elif args.bandit != None and args.multistart == None:
        my_source = f"./a.out_hypre_{args.equation}_nxmax{args.nmax}_nymax{args.nmax}_nzmax{args.nmax}_nxmin{args.nmin}_nymin{args.nmin}_nzmin{args.nmin}_nodes{args.nodes}_core32_ntask{args.ntask}_nrun{args.nrun}_{args.tuner}_bandit"
        save_path = f"./out_hypre_{args.equation}_nxmax{args.nmax}_nymax{args.nmax}_nzmax{args.nmax}_nxmin{args.nmin}_nymin{args.nmin}_nzmin{args.nmin}_nodes{args.nodes}_core32_ntask{args.ntask}_nrun{args.nrun}_{args.tuner}_bandit.pkl"
    elif args.multistart > 0 and args.bandit == None:
        my_source = f"./a.out_hypre_{args.equation}_nxmax{args.nmax}_nymax{args.nmax}_nzmax{args.nmax}_nxmin{args.nmin}_nymin{args.nmin}_nzmin{args.nmin}_nodes{args.nodes}_core32_ntask{args.ntask}_nrun{args.nrun}_{args.tuner}_multistart{args.multistart}"
        save_path = f"./out_hypre_{args.equation}_nxmax{args.nmax}_nymax{args.nmax}_nzmax{args.nmax}_nxmin{args.nmin}_nymin{args.nmin}_nzmin{args.nmin}_nodes{args.nodes}_core32_ntask{args.ntask}_nrun{args.nrun}_{args.tuner}_multistart{args.multistart}.pkl"
    else:
        raise NotImplementedError
    
    history = [] # history = [[task1, [runtime1, runtime2, ..., runtime30]], ]
    with open(my_source, "r") as f:
        line = f.readline()
        # print()
        i = 0
        while line:
            info = line.split()
            if len(info) > 0 and info[0] == '[-----' and info[-1] == '-----]':
                i += 1
                print(f"Find one runtime, i = {i}")
                runtime_cur = float(info[3])
                line = f.readline()
                line = f.readline()
                info = line.split()
                nx = int(info[0][2:-1])
                ny = int(info[1][:-1])
                nz = int(info[2][:-1])
                task_cur = (nx, ny, nz)
                idx = [task_cur in y for y in history]
                
                if idx == []:
                    history.append([task_cur, [runtime_cur]])
                elif max(idx) == True:
                    taskid = np.argmax(idx)
                    history[taskid][1].append(runtime_cur)
                else:
                    # add new task and first runtime
                    history.append([task_cur, [runtime_cur]])
            line = f.readline()

    # find min_runtime and nth
    for x in history:
        nth = np.argmin(x[1])
        Oopt = x[1][nth]
        x.append([f'{Oopt} nth  {nth}'])
    
    print(history)
    print(len(history))
    pickle.dump(history, open(save_path, "wb"))  

    
    
if __name__ == "__main__":
    main(parse_args())