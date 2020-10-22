import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmax', type=int, default=100, help='maximum discretization size')
    parser.add_argument('--nmin', type=int, default=10, help='minimum discretization size')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument("--equation", type=str, default="Poisson", help ='type of PDE to solve')
    parser.add_argument('--multistart', type=int, default=None, help='number of model restarts')
    parser.add_argument('--bandit', type=int, default=None, help='use bandit strategy in HpBandster or not')
    parser.add_argument('--tuner4', type=int, default=0, help='4 tuners or the usual 3 tuners')
    parser.add_argument('--rerun', type=int, default=0, help='rerun id')
    return parser.parse_args()

def get_results_from_line(line):
    info = line.split("\t")
    tuner = info[0]
    results = list(map(lambda x: float(x.split()[0]), info[1:]))
    return tuner, results

def main(args):
    print(args.tuner4)
    summary = []
    if args.tuner4 == True and args.multistart > 0 :
        if args.rerun == 0:
            my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}_tuner4.txt"
            save_path = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}_tuner4.pkl"
        else:
            my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}_tuner4_rerun{args.rerun}.txt"
            save_path = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}_tuner4_rerun{args.rerun}.pkl"
    elif args.multistart == None and args.bandit == None:
        my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}.txt"
        save_path = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}.pkl"
    elif args.bandit != None and args.multistart == None:
        my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_bandit.txt"
        save_path = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_bandit.pkl"
    elif args.multistart > 0 and args.bandit == None:
        my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}.txt"
        save_path = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}.pkl"
    else:
        raise NotImplementedError
    
    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split()
            if info[0] == 'Task':
                task_id = int(info[1][:-1])
                nx = int(info[-3][1:-1])
                ny = int(info[-2][:-1])
                nz = int(info[-1][:-1])
                info = f.readline().split()
                runs = [int(x) for x in info[3:]]
                line = f.readline()
                results = []
                while line.strip() != "":
                    results.append(get_results_from_line(line))
                    line = f.readline()
                summary.append((task_id, (nx, ny, nz), runs, results, nx*ny*nz))
            line = f.readline()
    # print(summary)
    # sort summary wrt nx*ny*nz
    summary = sorted(summary, key=itemgetter(4))
    pickle.dump(summary, open(save_path, "wb"))  
    # print()
    print(summary)

if __name__ == "__main__":
    main(parse_args())