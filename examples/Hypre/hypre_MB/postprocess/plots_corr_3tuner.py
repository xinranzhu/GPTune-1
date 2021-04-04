import os
import os.path as osp
import argparse
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', type=int, default=3, help='eta value in bandit structure')
    parser.add_argument('--bmax', type=int, default=100, help='maximum budget in bandit structure')
    parser.add_argument('--bmin', type=int, default=10, help='minimum budget in bandit structure')
    parser.add_argument('--ntask', type=str, default=None, help='number of tasks')
    parser.add_argument('--expid', type=str, default=None, help='experiment id')
    parser.add_argument('--width', type=float, default=0.05, help='experiment id')
    parser.add_argument('--tuner', type=int, default=3, help='number of tuners to compare with')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./data/corr_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}_ntask{args.ntask}_expid{args.expid}.pkl"
    return my_source

def gen_savepath(args, figurename, src=None):
    my_source = gen_source(args)    
    filename = os.path.splitext(os.path.basename(my_source))[0]
    my_path = os.path.join("./plots", f"{figurename}_{filename}.pdf")
    return my_path

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    tasks = results_summary[0]

    GPTune_MB_data = results_summary[1]
    Hp_data = results_summary[2]
    GPTune_data = results_summary[3]

    if args.tuner == 3:
        # absolute performance 
        x1 = GPTune_MB_data[1]
        x2 = Hp_data[1]
        x3 = GPTune_data[1]
        Xset = [x1, x2, x3]

        # relative performance
        Rset = np.array([x1, x2, x3])
        Rmin = np.min(Rset, axis=0)
        Rset /= np.expand_dims(Rmin, 0)

        # std of absolute performance
        s1 = GPTune_MB_data[2]
        s2 = Hp_data[2]
        s3 = GPTune_data[2]
        Sset = [s1, s2, s3]
    elif args.tuner == 2:
        # absolute performance 
        x1 = GPTune_MB_data[1]
        x3 = GPTune_data[1]
        Xset = [x1, x3]

        # relative performance
        Rset = np.array([x1, x3])
        Rmin = np.min(Rset, axis=0)
        Rset /= np.expand_dims(Rmin, 0)

        # std of absolute performance
        s1 = GPTune_MB_data[2]
        s3 = GPTune_data[2]
        Sset = [s1, s3]
    else:
        raise NotImplementedError()

    return Xset, Rset, Sset



def errorbar_plot(args, Xset, Rset, Sset):
    # ntask = args.ntask
    ntask = 10
    def single_errorbar(ax, x, s, tuner, colorcode, p=None):
        label=f'{tuner}, {p} wins' if p != None else f'{tuner}'
        ax.errorbar(np.arange(1,ntask+1), x, yerr = s, 
                    fmt='o', capsize=3, color=colorcode, label=label)

    if args.tuner == 3:
        p12 = len([x for x,y in zip(Xset[0],Xset[1]) if x <= y])
        p13 = len([x for x,y in zip(Xset[0],Xset[2]) if x <= y])
    
        plt.clf()
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(10, 4))
    
        single_errorbar(axs[0], Xset[1], Sset[1], 'HpBandster', '#ff7f0e')
        single_errorbar(axs[1], Xset[2], Sset[2], 'GPTune',     '#2ca02c')
        single_errorbar(axs[0], Xset[0], Sset[0], 'GPTuneBand', '#C33734', p=p12)  
        single_errorbar(axs[1], Xset[0], Sset[0], 'GPTuneBand', '#C33734', p=p13)    
        
        axs[0].set_xlabel('Task ID')
        axs[1].set_xlabel('Task ID')
        axs[0].set_ylabel('Optimal Hypre Time')
        axs[0].legend(fontsize=8)
        axs[1].legend(fontsize=8)

        figurename = "errorbar"
        savepath = gen_savepath(args, figurename)
        fig.savefig(savepath)
        print("Figure saved:, ", savepath)
    elif args.tuner == 2:
        p12 = len([x for x,y in zip(Xset[0],Xset[1]) if x <= y])
    
        plt.clf()
        plt.errorbar(np.arange(1,ntask+1), Xset[1], yerr = Sset[1], 
                    fmt='o', capsize=3, color='#2ca02c', label='GPTune')
        plt.errorbar(np.arange(1,ntask+1), Xset[0], yerr = Sset[0], 
                    fmt='o', capsize=3, color='#C33734', label=f'GPTuneBand, {p12} wins')
        plt.xlabel('Task ID')
        plt.ylabel('Optimal Hypre Time')
        plt.legend(fontsize=8)
        
        figurename = "errorbar"
        savepath = gen_savepath(args, figurename)
        plt.savefig(savepath)
        print("Figure saved:, ", savepath)
        
def autolabel(rects, ax):
    # """Attach a text label above each bar in *rects*, displaying its height."""
        for i in range(len(rects[0])):
            cur_height = rects[0][i]
            if cur_height > 0:
                ax.annotate(f'{cur_height:.2f}',
                            xy=((rects[1][i] + rects[1][i+1])/2, cur_height),
                            ha='center', va='bottom', fontsize=5)

def dist_relative(args, Rset):
    # ntask = args.ntask
    ntask = 10
    
    def single_hist(data, ax, tuner, colorcode):
        width = args.width
        rects = ax.hist(data, bins=np.arange(1, width+np.ceil(max(data)*100)/100, width), 
                        weights=np.ones(ntask) / ntask, 
                        label=f'{tuner}, mean={np.mean(data):.2f}, median={np.median(data):.2f}',
                        color=colorcode)
        autolabel(rects, ax)
        ax.legend(fontsize=8)
        ax.set_ylim(0,1)

    if args.tuner == 3:
        plt.clf()
        fig, axs = plt.subplots(nrows=3, ncols=1, sharey=True, sharex=True, figsize=(5.5,5))
        single_hist(Rset[0], axs[0], 'GPTuneBand', '#C33734')
        single_hist(Rset[1], axs[1], 'HpBandster', '#ff7f0e')
        single_hist(Rset[2], axs[2], 'GPTune',     '#2ca02c')

        figurename = "hist_relative"
        savepath = gen_savepath(args, figurename)
        fig.savefig(savepath)
        print("Figure saved:, ", savepath)
    elif args.tuner == 2:
        plt.clf()
        fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True, figsize=(5.5,5))
        single_hist(Rset[0], axs[0], 'GPTuneBand', '#C33734')
        single_hist(Rset[1], axs[1], 'GPTune',     '#2ca02c')

        figurename = "hist_relative"
        savepath = gen_savepath(args, figurename)
        fig.savefig(savepath)
        print("Figure saved:, ", savepath)
        
def main(args):
    Xset, Rset, Sset = data_process(args)
    errorbar_plot(args, Xset, Rset, Sset)
    dist_relative(args, Rset)

if __name__ == "__main__":
    main(parse_args())
    
    