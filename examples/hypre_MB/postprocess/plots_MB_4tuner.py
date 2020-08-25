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
    parser.add_argument('--tuner', type=int, default=4, help='minimum budget in bandit structure')
    return parser.parse_args()

def gen_source(args):
    my_source = f"./data_4tuner/MLA_MB_amin{args.amin}_cmin{args.cmin}_amax{args.amax}_cmax{args.cmax}_ntask{args.ntask}_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}.pkl"
    return my_source

def gen_savepath(args, figurename, src=None):
    my_source = gen_source(args)    
    filename = os.path.splitext(os.path.basename(my_source))[0]
    my_path = os.path.join("./plots_4tuner", f"{figurename}_{filename}.pdf")
    # my_path = f"./plots/MLA_MB_{figurename}_amin{args.amin}_cmin{args.cmin}_amax{args.amax}_cmax{args.cmax}_ntask{args.ntask}_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}.pdf"
    return my_path

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    tasks = results_summary[0]

    GPTune_MB_data = results_summary[1]
    Hp_data = results_summary[2]
    GPTune_data = results_summary[3]
    Op_data = results_summary[4]

    x1 = GPTune_MB_data[1]
    x2 = Hp_data[1]
    x3 = GPTune_data[1]
    x4 = Op_data[1]
    s1 = GPTune_MB_data[2]
    s2 = Hp_data[2]
    s3 = GPTune_data[2]
    s4 = Op_data[2]
    x21 = np.array(x2)/np.array(x1)
    x31 = np.array(x3)/np.array(x1)
    x41 = np.array(x4)/np.array(x1)
    p12 = len([x for x,y in zip(x1,x2) if x <= y])
    p13 = len([x for x,y in zip(x1,x3) if x <= y])
    p14 = len([x for x,y in zip(x1,x4) if x <= y])
    return x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14

def errorbar_plot(args, x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14):
    plt.clf()
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True, figsize=(12, 10))
    axs[0,0].errorbar(np.arange(1,21), x1, yerr = s1, 
                fmt='o', capsize=3, color='#C33734', label='GPTuneBand')
    axs[0,1].errorbar(np.arange(1,21), x1, yerr = s1, 
                fmt='o', capsize=3, color='#C33734', label=f'GPTuneBand, {p12} wins')
    axs[1,0].errorbar(np.arange(1,21), x1, yerr = s1, 
                fmt='o', capsize=3, color='#C33734', label=f'GPTuneBand, {p13} wins')
    axs[1,1].errorbar(np.arange(1,21), x1, yerr = s1, 
                fmt='o', capsize=3, color='#C33734', label=f'GPTuneBand, {p14} wins')

    axs[0,1].errorbar(np.arange(1,21), x2, yerr = s2, 
                fmt='d', capsize=3, color='#ff7f0e', label='HpBandster')
    axs[1,0].errorbar(np.arange(1,21), x3, yerr = s3, 
                fmt='v', capsize=3, color='#2ca02c', label='GPTune')
    axs[1,1].errorbar(np.arange(1,21), x4, yerr = s4, 
                fmt='*', capsize=3, color='#1f77b4', label='OpenTuner')

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

def ratiobars_plot(args, x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14):
    plt.clf()
    x = np.arange(1, args.ntask+1)
    width=0.8/3
    fig, ax = plt.subplots()
    ax.bar(x-width, x21, width, label=f'HpBandster/GPTuneBand, {p12}(>=1)', color='#ff7f0e')
    ax.bar(x, x31, width, label=f'GPTune/GPTuneBand, {p13}(>=1)', color='#2ca02c')
    ax.bar(x+width, x41, width, label=f'OpenTuner/GPTuneBand, {p14}(>=1)', color='#1f77b4')
    ax.plot([0, args.ntask+1], [1, 1], c='black', linestyle=':')
    ax.set_ylabel('Ratio of best performance')
    ax.set_xlabel('Task ID')  
    ax.legend(fontsize=8)
    fig.tight_layout()
    
    figurename = "ratiobars"
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

def hist_plot(args, x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14):
    ntask = args.ntask
    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    rects1 = ax1.hist(x21, bins=np.arange(0, int(np.ceil(max(x21)))+1, 0.25), weights=np.ones(ntask) / ntask, 
             label=f'HpBandster/GPTuneBand', color='#ff7f0e')
    rects2 = ax2.hist(x31, bins=np.arange(0, int(np.ceil(max(x31)))+1, 0.25) , weights=np.ones(ntask) / ntask, 
                label=f'GPTune/GPTuneBand', color='#2ca02c')
    rects3 = ax3.hist(x41, bins=np.arange(0, int(np.ceil(max(x41)))+1, 0.25) , weights=np.ones(ntask) / ntask, 
                label=f'OpenTuner/GPTuneBand', color='#1f77b4')
    ax1.plot([1, 1], [0, 1], c='black', linestyle=':')
    ax2.plot([1, 1], [0, 1], c='black', linestyle=':')
    ax3.plot([1, 1], [0, 1], c='black', linestyle=':')
    ax1.set_xticks(np.arange(0, int(np.ceil(max(x21)))+1, step=0.5))
    ax2.set_xticks(np.arange(0, int(np.ceil(max(x31)))+1, step=0.5))
    ax3.set_xticks(np.arange(0, int(np.ceil(max(x31)))+1, step=0.5))
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    ax3.legend(fontsize=8)
    autolabel(rects1, ax1)
    autolabel(rects2, ax2)
    autolabel(rects3, ax3)
    figurename = "hist"
    savepath = gen_savepath(args, figurename)
    fig.savefig(savepath)
    print("Figure saved:, ", savepath)

def hist_plot_KDE(args, x21, x31, x41):
    plt.clf()
    sns.set(color_codes=True)
    sns.distplot(x21, label=f"HpBandster/GPTuneBand, mean={np.mean(x21):.2f}, median={np.median(x21):.2f}", color="#ff7f0e")
    sns.distplot(x31, label=f"GPTune/GPTuneBand, mean={np.mean(x31):.2f}, median={np.median(x31):.2f}", color="#2ca02c")
    sns.distplot(x41, label=f"OpenTuner/GPTuneBand, mean={np.mean(x41):.2f}, median={np.median(x41):.2f}", color="#1f77b4")
    plt.legend(fontsize=8)
    figurename = "hist_KDE"
    savepath = gen_savepath(args, figurename)
    plt.savefig(savepath)
    print("Figure saved:, ", savepath)
    plt.legend(fontsize=8)
    
def main(args):
    x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14 = data_process(args)
    errorbar_plot(args, x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14 )
    ratiobars_plot(args, x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14)
    hist_plot(args, x1, x2, x3, x4, s1, s2, s3, s4, x21, x31, x41, p12, p13, p14 )
    hist_plot_KDE(args, x21, x31, x41)
    
if __name__ == "__main__":
    main(parse_args())
    
    


    
    