import os
import os.path as osp
import argparse
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmax', type=int, default=100, help='maximum discretization size')
    parser.add_argument('--nmin', type=int, default=10, help='minimum discretization size')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument("--equation", type=str, default="poisson", help ='type of PDE to solve')
    parser.add_argument('--nrun', type=int, default=10, help='number of runs')
    parser.add_argument('--xtype', type=str, default='time', help='type of x-axis')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./data_MLA_history/exp_hypre_history_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_nrun{args.nrun}.pkl"
    return my_source

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    GPTune_results = results_summary[0]
    OpenTuner_results = results_summary[1]
    HpBandster_results = results_summary[2]
    assert GPTune_results[0] == 'GPTune'
    assert OpenTuner_results[0] == 'OpenTuner'
    assert HpBandster_results[0] == 'HpBandster'
    return GPTune_results, OpenTuner_results, HpBandster_results

def historical_best(data):
    for i in range(len(data)-1):
        data[i+1] = data[i] if data[i] < data[i+1] else data[i+1]

def total_time_estimate(data):
    data2 = data.copy()
    for i in range(len(data2)-1):
        data2[i+1] += data2[i] 
    return data2

# i: sorted i-th task
def plot_single(args, i, GPTune_results, OpenTuner_results, HpBandster_results, ax=None, singleplot=False):
    my_source = gen_source(args)
    ntask = args.ntask
    nrun = args.nrun
    GPTune_i = GPTune_results[1:][i]
    OpenTuner_i = OpenTuner_results[1:][i]
    HpBandster_i = HpBandster_results[1:][i]
    assert GPTune_i[0] == OpenTuner_i[0]
    assert GPTune_i[0] == HpBandster_i[0]
    assert GPTune_i[1] == OpenTuner_i[1]
    assert GPTune_i[1] == HpBandster_i[1]
    taskID = GPTune_i[0]
    tasksize = GPTune_i[1] # (nx, ny, nz)
    GPTune_hist = GPTune_i[2]
    OpenTuner_hist = OpenTuner_i[2]
    HpBandster_hist = HpBandster_i[2] 

    GPTune_timehist = total_time_estimate(GPTune_hist)
    OpenTuner_timehist = total_time_estimate(OpenTuner_hist)
    HpBandster_timehist = total_time_estimate(HpBandster_hist)

    # best historical 
    historical_best(GPTune_hist)
    historical_best(OpenTuner_hist)
    historical_best(HpBandster_hist)

    if ax == None:
        plt.clf()
        ax = plt.gca()

    if args.xtype == "time":
        ax.plot(GPTune_timehist, GPTune_hist, color='#2ca02c') # green 
        ax.plot(OpenTuner_timehist, OpenTuner_hist, color='#1f77b4') # blue
        ax.plot(HpBandster_timehist, HpBandster_hist, color='#ff7f0e') # orange
        # ax.set_xlabel('Tuning time', fontsize=8)
        # ax.set_ylabel('Historical best runtime', fontsize=8)
        # ax.legend(fontsize=8)
    elif args.xtype == "iter":
        ax.plot(np.arange(len(GPTune_hist)), GPTune_hist, color='#2ca02c')
        ax.plot(np.arange(len(OpenTuner_hist)), OpenTuner_hist, color='#1f77b4')
        ax.plot(np.arange(len(HpBandster_hist)), HpBandster_hist, color='#ff7f0e')
    else:
        raise NotImplementedError()

    ax.grid()
    ax.set_title(f'(nx, ny, nz) = {tasksize}', fontsize=9)
    if singleplot:
        filename = os.path.splitext(os.path.basename(my_source))[0]
        plt.savefig(os.path.join("./plots_MLA_history", f"singleplot_{filename}_nrun{nrun}_task{i}.pdf"))

def plot_group(args, GPTune_results, OpenTuner_results, HpBandster_results):
    if args.ntask != 30:
        raise NotImplementedError()
    my_source = gen_source(args)

    plt.clf()
    fig, ax = plt.subplots(10, 3, figsize=(10, 25))
    # fig, ax = plt.subplots(10, 3)
    for i in range(10):
        for j in range(3):
            taskID = 3*i+j
            # ax[i, j].plot(np.arange(5), np.random.rand(5))
            plot_single(args, taskID, GPTune_results, OpenTuner_results, HpBandster_results, ax=ax[i,j])
    
    equation = "Convection-diffusion equation" if args.equation == "convdiff" else "Poisson equation"
    fig.suptitle(f"{equation}, nrun = {args.nrun}", fontsize=16)
    plt.subplots_adjust(hspace=0.4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    filename = os.path.splitext(os.path.basename(my_source))[0]
    plt.savefig(os.path.join("./plots_MLA_history/", f"{filename}_ntask{args.ntask}_nrun{args.nrun}_{args.xtype}.pdf"))




def main(args):
    GPTune_results, OpenTuner_results, HpBandster_results = data_process(args)

    # plot the i-th tasks (sorted)
    # plot_single(args, 0, GPTune_results, OpenTuner_results, HpBandster_results, singleplot=True)

    # plot all 30 tasks 
    plot_group(args, GPTune_results, OpenTuner_results, HpBandster_results)
    
if __name__ == "__main__":
    main(parse_args())