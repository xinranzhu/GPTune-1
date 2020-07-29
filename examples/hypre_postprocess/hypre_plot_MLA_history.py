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
    parser.add_argument("--equation", type=str, default="Poisson", help ='type of PDE to solve')
    parser.add_argument('--nrun', type=int, default=10, help='number of runs')
    parser.add_argument('--xtype', type=str, default='time', help='type of x-axis')
    parser.add_argument("--multistart", type=int, default=0, help ='number of model restarts')
    parser.add_argument("--ratio2best", type=str, default="None", help ='plot ratio to anytime best or alltime best')
    parser.add_argument('--taskid', type=int, default=0, help='single plot of the given task, sorted')
    return parser.parse_args()

def gen_source(args):
    if args.multistart == 0:
        my_source = f'./data_MLA_history/exp_hypre_history_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_nrun{args.nrun}.pkl'
    else:
        my_source = f'./data_MLA_history/exp_hypre_history_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_nrun{args.nrun}_multistart{args.multistart}.pkl'
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
    filename = os.path.splitext(os.path.basename(my_source))[0]
    ntask = args.ntask
    nrun = args.nrun
    GPTune_i = GPTune_results[1:][i]
    OpenTuner_i = OpenTuner_results[1:][i]
    HpBandster_i = HpBandster_results[1:][i]
    # assert GPTune_i[0] == OpenTuner_i[0]
    # assert GPTune_i[0] == HpBandster_i[0]
    assert GPTune_i[1] == OpenTuner_i[1]
    assert GPTune_i[1] == HpBandster_i[1]
    taskID = GPTune_i[0]
    tasksize = GPTune_i[1] # (nx, ny, nz)
    
    # make sure all history same length
    GPTune_hist = GPTune_i[2][:nrun]
    OpenTuner_hist = OpenTuner_i[2][:nrun]
    HpBandster_hist = HpBandster_i[2][:nrun]
    if len(OpenTuner_hist) == nrun - 1:
            GPTune_hist = GPTune_hist[:-1]
            HpBandster_hist = HpBandster_hist[:-1]
        
    GPTune_timehist = total_time_estimate(GPTune_hist)
    OpenTuner_timehist = total_time_estimate(OpenTuner_hist)
    HpBandster_timehist = total_time_estimate(HpBandster_hist)

    print(f"Original history for check, task id = {taskID}, tasksize = {tasksize}")
    print("GPTune, ", GPTune_hist)
    print("OpenTuner, ", OpenTuner_hist)
    print("HpBandster, ", HpBandster_hist)
    print("")
    
    # historical best
    historical_best(GPTune_hist)
    historical_best(OpenTuner_hist)
    historical_best(HpBandster_hist)
    
    max_height = max(GPTune_hist[0], OpenTuner_hist[0], HpBandster_hist[0])
    min_height = min(GPTune_hist[-1], OpenTuner_hist[-1], HpBandster_hist[-1])
    if ax == None:
        plt.clf()
        ax = plt.gca()

    if args.ratio2best == "None":
        if args.xtype == "time":
            ax.plot(GPTune_timehist, GPTune_hist, color='#2ca02c') # green 
            ax.plot(OpenTuner_timehist, OpenTuner_hist, color='#1f77b4') # blue
            ax.plot(HpBandster_timehist, HpBandster_hist, color='#ff7f0e') # orange
            ax.plot([GPTune_timehist[int(nrun/2)], GPTune_timehist[int(nrun/2)]], [min_height, max_height], c='black', linestyle=':')
        elif args.xtype == "iter":
            ax.plot(np.arange(len(GPTune_hist)), GPTune_hist, color='#2ca02c')
            ax.plot(np.arange(len(OpenTuner_hist)), OpenTuner_hist, color='#1f77b4')
            ax.plot(np.arange(len(HpBandster_hist)), HpBandster_hist, color='#ff7f0e')
            ax.plot([int(nrun/2), int(nrun/2)], [min_height, max_height], c='black', linestyle=':')
        else:
            raise NotImplementedError()
    else: # need to compute ratio2best
        if args.ratio2best == "anytime":
            anytime_best = np.minimum(np.minimum(GPTune_hist, OpenTuner_hist), HpBandster_hist)
            GPTune_to_best = GPTune_hist/anytime_best
            OpenTuner_to_best = OpenTuner_hist/anytime_best
            HpBandster_to_best = HpBandster_hist/anytime_best
        elif args.ratio2best == "alltime":
            alltime_best = min(GPTune_hist + OpenTuner_hist + HpBandster_hist)
            GPTune_to_best = list(map(lambda x: x/alltime_best, GPTune_hist))
            OpenTuner_to_best = list(map(lambda x: x/alltime_best, OpenTuner_hist))
            HpBandster_to_best = list(map(lambda x: x/alltime_best, HpBandster_hist))
        else:
            raise NotImplementedError()
            
        max_height = max(np.maximum(np.maximum(GPTune_to_best, OpenTuner_to_best), HpBandster_to_best))
        if args.xtype == "time":
            ax.plot(GPTune_timehist[:len(GPTune_hist)], GPTune_to_best, color='#2ca02c') # green 
            ax.plot(OpenTuner_timehist[:len(OpenTuner_hist)], OpenTuner_to_best, color='#1f77b4') # blue
            ax.plot(HpBandster_timehist[:len(HpBandster_hist)], HpBandster_to_best, color='#ff7f0e') # orange
            ax.plot([GPTune_timehist[int(nrun/2)], GPTune_timehist[int(nrun/2)]], [1., max_height], c='black', linestyle=':')
        elif args.xtype == "iter":
            ax.plot(np.arange(len(GPTune_hist)), GPTune_to_best, color='#2ca02c')
            ax.plot(np.arange(len(OpenTuner_hist)), OpenTuner_to_best, color='#1f77b4')
            ax.plot(np.arange(len(HpBandster_hist)), HpBandster_to_best, color='#ff7f0e')
            ax.plot([int(nrun/2), int(nrun/2)], [1., max_height], c='black', linestyle=':')
        else:
            raise NotImplementedError()
    ax.grid()
    ax.set_title(f'(nx, ny, nz) = {tasksize}', fontsize=9)
    if singleplot:
        ax.set_xlabel('Tuning time', fontsize=8)
        if args.ratio2best == "None":
            ax.set_ylabel('Historical best runtime', fontsize=8)
        elif args.ratio2best == "anytime":
            ax.set_ylabel('Ratio to anytime best', fontsize=8)
        elif args.ratio2best == "alltime":
            ax.set_ylabel('Ratio to alltime best', fontsize=8)
        else:
            raise NotImplementedError()
        ax.legend(fontsize=8)
        savepath = os.path.join("./plots_MLA_history", f"singleplot_{filename}_nrun{nrun}_task{i}_{args.xtype}_best{args.ratio2best}.pdf")    
        plt.savefig(savepath)
        print("Figure saved: ", savepath)
    if args.ratio2best != "None":
        return np.mean(GPTune_to_best[int(nrun/2):]), np.mean(OpenTuner_to_best[int(nrun/2):]), np.mean(HpBandster_to_best[int(nrun/2):])
    
def plot_group(args, GPTune_results, OpenTuner_results, HpBandster_results):
    GPTune_meanset = []
    OpenTuner_meanset = []
    HpBandster_meanset = []
    
    if args.ntask != 30:
        raise NotImplementedError()
    my_source = gen_source(args)
    filename = os.path.splitext(os.path.basename(my_source))[0]

    plt.clf()
    fig, ax = plt.subplots(10, 3, figsize=(10, 25))
    # fig, ax = plt.subplots(10, 3)
    for i in range(10):
        for j in range(3):
            taskID = 3*i+j
            # ax[i, j].plot(np.arange(5), np.random.rand(5))
            GPTune_taskID, OpenTuner_taskID, HpBandster_taskID = plot_single(args, taskID, GPTune_results, OpenTuner_results, HpBandster_results, ax=ax[i,j])
            GPTune_meanset.append(GPTune_taskID)
            OpenTuner_meanset.append(OpenTuner_taskID)
            HpBandster_meanset.append(HpBandster_taskID)            
    equation = "Convection-diffusion equation" if args.equation == "convdiff" else "Poisson equation"
    fig.suptitle(f"{equation}, nrun = {args.nrun}", fontsize=16)
    plt.subplots_adjust(hspace=0.4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    
    
    savepath = os.path.join("./plots_MLA_history/", f"{filename}_{args.xtype}_best{args.ratio2best}.pdf")
    plt.savefig(savepath)
    print("Figure saved:", savepath)
    return GPTune_meanset, OpenTuner_meanset, HpBandster_meanset
    
def autolabel(rects, ax):
    # """Attach a text label above each bar in *rects*, displaying its height."""
        for i in range(len(rects[0])):
            cur_height = rects[0][i]
            if cur_height > 0:
                ax.annotate(f'{cur_height:.2f}',
                            xy=((rects[1][i] + rects[1][i+1])/2, cur_height),
                            ha='center', va='bottom', fontsize=5)

def plot_group_meanset_hist(data1, data2, data3, args):
    my_source = gen_source(args)
    filename = os.path.splitext(os.path.basename(my_source))[0]
    nrun = args.nrun
    savepath = os.path.join("./plots_MLA_history", f"hist_{filename}_best{args.ratio2best}.pdf")
    assert len(data1) == len(data2) 
    assert len(data1) == len(data3)
    ntask = len(data1)
    
    plt.clf()
    maxrange = np.ceil(4*max(data1+data2+data3))/4
    print(f"maxrange = {maxrange}")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8))
    rects1 = ax1.hist(data1, bins=np.arange(1, maxrange+0.5, 0.25), weights=np.ones(ntask) / ntask, 
                      label=f'GPTune, mean={np.mean(data1):.2f}, median={np.median(data1):.2f}', color='#2ca02c')
    rects2 = ax2.hist(data2, bins=np.arange(1, maxrange+0.5, 0.25) , weights=np.ones(ntask) / ntask, 
                      label=f'OpenTuner, mean={np.mean(data2):.2f}, median={np.median(data2):.2f}', color='#1f77b4')    
    rects3 = ax3.hist(data3, bins=np.arange(1, maxrange+0.5, 0.25) , weights=np.ones(ntask) / ntask, 
                      label=f'HpBandster, mean={np.mean(data3):.2f}, median={np.median(data3):.2f}', color='#ff7f0e')   
    # ax1.set_xticks(np.arange(1, np.ceil(2*maxrange)/2, step=0.5))
    # ax2.set_xticks(np.arange(1, np.ceil(2*maxrange)/2, step=0.5))
    # ax3.set_xticks(np.arange(1, np.ceil(2*maxrange)/2, step=0.5))
    ax1.set_xlim(1, np.ceil(maxrange))
    ax2.set_xlim(1, np.ceil(maxrange))
    ax3.set_xlim(1, np.ceil(maxrange))
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax3.legend(fontsize=8)
    autolabel(rects1, ax1)
    autolabel(rects2, ax2)
    autolabel(rects3, ax3)

    # equation_name = "Poisson"
    # if args.equation == "convdiff":
    #     equation_name = "Convection-diffusion"
    # ax1.set_title(f'{equation_name}, [nx, ny, nz] in [{args.nmin}, {args.nmax}], nrun = {nrun}')
    ax1.set_ylabel('Density')
    ax2.set_ylabel('Density')
    ax3.set_ylabel('Density')
    ax1.set_xlabel('mean(ratio2best)')
    ax2.set_xlabel('mean(ratio2best)')
    ax3.set_xlabel('mean(ratio2best)')
    fig.tight_layout()
    fig.savefig(savepath)
    print("Figure saved:, ", savepath)


def main(args):
    GPTune_results, OpenTuner_results, HpBandster_results = data_process(args)

    if args.taskid > 0:
        # plot the i-th tasks (sorted)
        if args.ratio2best != "None":
            GPTune_taskID, OpenTuner_taskID, HpBandster_taskID = plot_single(args, args.taskid, GPTune_results, OpenTuner_results, HpBandster_results, singleplot=True)
            print(f"Single task: task id {args.taskid} (sorted id)")
            print(f"GPTune_taskID = {GPTune_taskID}")
            print(f"OpenTuner_taskID = {OpenTuner_taskID}") 
            print(f"HpBandster_taskID = {HpBandster_taskID}") 
        else:
            plot_single(args, args.taskid, GPTune_results, OpenTuner_results, HpBandster_results, singleplot=True)
    else:
        if args.ratio2best != "None":
            # plot all 30 tasks 
            GPTune_meanset, OpenTuner_meanset, HpBandster_meanset = plot_group(args, GPTune_results, OpenTuner_results, HpBandster_results)
            print(f"GPTune_meanset = {GPTune_meanset}")
            print(f"OpenTuner_meanset = {OpenTuner_meanset}") 
            print(f"HpBandster_meanset = {HpBandster_meanset}") 
            plot_group_meanset_hist(GPTune_meanset, OpenTuner_meanset, HpBandster_meanset, args)
        else:
            plot_group(args, GPTune_results, OpenTuner_results, HpBandster_results)

if __name__ == "__main__":
    main(parse_args())