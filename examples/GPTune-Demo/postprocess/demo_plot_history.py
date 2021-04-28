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
    parser.add_argument('-ntask', type=int, default=1, help='number of tasks')
    parser.add_argument("-Nloop", type=int, default=1, help ='number of bandit loops')
    parser.add_argument('-expid', type=str, default='0')
    parser.add_argument('-fmin', type=float, default=0.)
    return parser.parse_args()

def gen_source(args):
    my_source = f'./data/demo_ntask{args.ntask}_Nloop{args.Nloop}_expid{args.expid}.pkl'
    return my_source

def data_process(args):
    my_source = gen_source(args)
    results_summary = pickle.load(open(my_source, "rb"))
    GPTuneBand_results = results_summary[0]
    GPTune_results = results_summary[1]
    HpBandster_results = results_summary[2]
    TPE_results = results_summary[3]
    OpenTuner_results = results_summary[4]
    print(GPTuneBand_results)
    assert GPTuneBand_results[0] == 'GPTuneBand'
    assert GPTune_results[0] == 'GPTune'
    assert HpBandster_results[0] == 'hpbandster'
    assert TPE_results[0] == 'TPE'
    assert OpenTuner_results[0] == 'opentuner'
    
    colors = ['#C33734', '#2ca02c', '#ff7f0e', '#9467bd', '#1f77b4']
    linestyles = ['solid', 'dashed', 'dashdot', 'dashdot', 'dotted']
    data_summary = []
    tuners = []
    for item in results_summary:
        tuners.append(item[0])
        data_summary.append(item[1][2])
    return data_summary, tuners, colors, linestyles

def historical_best(data):
    for i in range(len(data)-1):
        data[i+1] = data[i] if data[i] < data[i+1] else data[i+1]
    return data

# i: sorted i-th task
def plot_single(args, data_summary, tuners, colors, linestyles):
    my_source = gen_source(args)
    filename = os.path.splitext(os.path.basename(my_source))[0]
    print(filename)
    
    # historical best
    # print("Original data summary")
    # print(data_summary)
    data_summary_new = []
    for i,data in enumerate(data_summary):
        print(tuners[i])
        print(data)
        data_summary_new.append([data[0], historical_best(data[1])])
    # print(data_summary_new)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data_summary_new[0][0], 
            args.fmin*np.ones(len(data_summary_new[0][0])), 'k--', label="True")
    for i,data in enumerate(data_summary_new):
        ax.plot(data[0], data[1], color=colors[i], label=tuners[i], linestyle=linestyles[i])

    ax.legend(fontsize=8)
    savepath = os.path.join("./plots", f"singleplot_{filename}.pdf")    
    plt.savefig(savepath)
    print("Figure saved: ", savepath)



def main(args):
    data_summary, tuners, colors, linestyles = data_process(args)
    plot_single(args, data_summary, tuners, colors, linestyles)

    
if __name__ == "__main__":
    main(parse_args())