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
    parser.add_argument('--amax', type=int, default=2, help='maximum of coeff_a')
    parser.add_argument('--amin', type=int, default=0, help='minimum of coeff_a')
    parser.add_argument('--cmax', type=int, default=2, help='maximum of coeff_c')
    parser.add_argument('--cmin', type=int, default=0, help='minimum of coeff_c')
    parser.add_argument('--ntask', type=int, default=20, help='number of tasks')
    parser.add_argument('--eta', type=int, default=3, help='eta value in bandit structure')
    parser.add_argument('--bmax', type=int, default=100, help='maximum budget in bandit structure')
    parser.add_argument('--bmin', type=int, default=10, help='minimum budget in bandit structure')
    parser.add_argument('--expid', type=str, default=None, help='experiment id')
    parser.add_argument('--width', type=float, default=0.05, help='experiment id')
    parser.add_argument('--tuner', type=int, default=4, help='minimum budget in bandit structure')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./data_5tuner/MLA_MB_amin{args.amin}_cmin{args.cmin}_amax{args.amax}_cmax{args.cmax}_ntask{args.ntask}_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}_expid{args.expid}.pkl"
    return my_source

def gen_savepath(args, figurename, src=None):
    my_source = gen_source(args)    
    filename = os.path.splitext(os.path.basename(my_source))[0]
    my_path = os.path.join("./plots_5tuner", f"{figurename}_{filename}.pdf")
    return my_path

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    tasks = results_summary[0]

    GPTune_MB_data = results_summary[1]
    Hp_data = results_summary[2]
    GPTune_data = results_summary[3]
    Op_data = results_summary[4]
    TPE_data = results_summary[5]

    # absolute performance 
    x1 = GPTune_MB_data[1]
    x2 = Hp_data[1]
    x3 = GPTune_data[1]
    x4 = Op_data[1]
    x5 = TPE_data[1]
    Xset = [x1, x2, x3, x4, x5]

    # relative performance
    Rset = np.array([x1, x2, x3, x4, x5])
    Rmin = np.min(Rset, axis=0)
    Rset /= np.expand_dims(Rmin, 0)

    # std of absolute performance
    s1 = GPTune_MB_data[2]
    s2 = Hp_data[2]
    s3 = GPTune_data[2]
    s4 = Op_data[2]
    s5 = TPE_data[2]
    Sset = [s1, s2, s3, s4, s5]

    # x21 = np.array(x2)/np.array(x1)
    # x31 = np.array(x3)/np.array(x1)
    # x41 = np.array(x4)/np.array(x1)
    # x51 = np.array(x5)/np.array(x1)

    # p12 = len([x for x,y in zip(x1,x2) if x <= y])
    # p13 = len([x for x,y in zip(x1,x3) if x <= y])
    # p14 = len([x for x,y in zip(x1,x4) if x <= y])

    return Xset, Rset, Sset



def errorbar_plot(args, Xset, Rset, Sset):
    ntask = args.ntask
    p12 = len([x for x,y in zip(Xset[0],Xset[1]) if x <= y])
    p13 = len([x for x,y in zip(Xset[0],Xset[2]) if x <= y])
    p14 = len([x for x,y in zip(Xset[0],Xset[3]) if x <= y])
    p15 = len([x for x,y in zip(Xset[0],Xset[4]) if x <= y])

    plt.clf()
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(12, 10))
    def single_errorbar(ax, x, s, tuner, colorcode, p=None):
        label=f'{tuner}, {p} wins' if p != None else f'{tuner}'
        ax.errorbar(np.arange(1,ntask+1), x, yerr = s, 
                    fmt='o', capsize=3, color=colorcode, label=label)

    single_errorbar(axs[0,0], Xset[1], Sset[1], 'HpBandster', '#ff7f0e')
    single_errorbar(axs[0,1], Xset[2], Sset[2], 'GPTune',     '#2ca02c')
    single_errorbar(axs[1,0], Xset[3], Sset[3], 'OpenTuner',  '#1f77b4')
    single_errorbar(axs[1,1], Xset[4], Sset[4], 'TPE',        '#9467bd')
    single_errorbar(axs[0,0], Xset[0], Sset[0], 'GPTuneBand', '#C33734', p=p12)  
    single_errorbar(axs[0,1], Xset[0], Sset[0], 'GPTuneBand', '#C33734', p=p13)    
    single_errorbar(axs[1,0], Xset[0], Sset[0], 'GPTuneBand', '#C33734', p=p14)    
    single_errorbar(axs[1,1], Xset[0], Sset[0], 'GPTuneBand', '#C33734', p=p15)    

    axs[1,0].set_xlabel('Task ID')
    axs[1,1].set_xlabel('Task ID')
    axs[0,0].set_ylabel('Optimal Hypre Time')
    axs[1,0].set_ylabel('Optimal Hypre Time')
    axs[0,0].legend(fontsize=8)
    axs[0,1].legend(fontsize=8)
    axs[1,0].legend(fontsize=8)
    axs[1,1].legend(fontsize=8)

    figurename = "errorbar"
    savepath = gen_savepath(args, figurename)
    fig.savefig(savepath)
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
    ntask = args.ntask
    plt.clf()
    fig, axs = plt.subplots(nrows=5, ncols=1, sharey=True, sharex=True, figsize=(6,10))

    def single_hist(data, ax, tuner, colorcode):
        width = args.width
        rects = ax.hist(data, bins=np.arange(1, width+np.ceil(max(data)*100)/100, width), 
                        weights=np.ones(ntask) / ntask, 
                        label=f'{tuner}, mean={np.mean(data):.2f}, median={np.median(data):.2f}',
                        color=colorcode)
        autolabel(rects, ax)
        ax.legend(fontsize=8)
        ax.set_ylim(0,1)

    single_hist(Rset[0], axs[0], 'GPTuneBand', '#C33734')
    single_hist(Rset[1], axs[1], 'HpBandster', '#ff7f0e')
    single_hist(Rset[2], axs[2], 'GPTune',     '#2ca02c')
    single_hist(Rset[3], axs[3], 'OpenTuner',  '#1f77b4')
    single_hist(Rset[4], axs[4], 'TPE',        '#9467bd')

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
    
    