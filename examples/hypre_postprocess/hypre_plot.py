import os
import os.path as osp
import argparse
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmax', type=int, default=100, help='maximum discretization size')
    parser.add_argument('--nmin', type=int, default=10, help='minimum discretization size')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument("--equation", type=str, default="poisson", help ='type of PDE to solve')
    parser.add_argument('--nrun', type=int, default=10, help='number of runs')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}.pkl"
    return my_source

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    ntask = len(results_summary)
    assert ntask == args.ntask
    OpenTunervsGPTune = []
    HpBanstervsGPTune = []
    for i in range(ntask):
        task_current = results_summary[i]
        idx_nrun = task_current[2].index(args.nrun)
        results_GPTune = task_current[3][0]
        results_OpenTuner = task_current[3][1]
        results_Hpbanster = task_current[3][2]
        assert results_GPTune[0] == "GPTune"
        assert results_OpenTuner[0] == "OpenTuner"
        assert results_Hpbanster[0] == "Hpbandster"
        OpenTunervsGPTune.append(results_OpenTuner[1][idx_nrun]/results_GPTune[1][idx_nrun])
        HpBanstervsGPTune.append(results_Hpbanster[1][idx_nrun]/results_GPTune[1][idx_nrun])
    
    return OpenTunervsGPTune, HpBanstervsGPTune

def plot(data1, data2, args):
    my_source = gen_source(args)
    assert len(data1) == len(data2) 
    nrun = args.nrun
    ntask = len(data1)
    
    p1 = len([x for x in data1 if x >= 1])
    p2 = len([x for x in data2 if x >= 1])

    # plot
    x = np.arange(1, ntask+1)
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - width/2, data1, width, label=f'OpenTuner/GPTune, {p1}')
    ax.bar(x + width/2, data2, width, label=f'HpBanster/GPTune, {p2}')
    ax.plot([0,ntask+1], [1, 1], c='black')
    ax.set_ylabel('ratio of best performance')
    ax.set_title(f'{args.equation}, [nx, ny, nz] in [{args.nmin}, {args.nmax}], nrun = {nrun}')
    ax.legend()
    fig.tight_layout()
    
    
    filename = os.path.splitext(os.path.basename(my_source))[0]
    fig.savefig(os.path.join("./plots", f"{filename}_nrun{nrun}.pdf"))

def main(args):
    OpenTunervsGPTune, HpBanstervsGPTune = data_process(args)
    print(OpenTunervsGPTune)
    print(HpBanstervsGPTune)
    plot(OpenTunervsGPTune, HpBanstervsGPTune, args)

if __name__ == "__main__":
    main(parse_args())