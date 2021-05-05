import os
import os.path as osp
import argparse
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
from operator import itemgetter
from argparse import Namespace

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='susy_10Kn')
    parser.add_argument('-ntask', type=int, default=1, help='number of tasks')
    parser.add_argument("-bmin", type=int, default=1, help ='minimum value for bandit budget')
    parser.add_argument("-bmax", type=int, default=8, help ='maximum value for bandit budget')
    parser.add_argument("-eta", type=int, default=2, help ='base value for bandit structure')
    parser.add_argument("-Nloop", type=int, default=1, help ='number of bandit loops')
    parser.add_argument('-baseline', type=float, default=None)
    parser.add_argument('-explist', nargs='+', help='a list of repeated experiment for error bar plots', required=True)
    parser.add_argument('-deleted_tuners', nargs='+', help='a list of tuners not plotting', default=None)
    parser.add_argument('-taskid', type=int, default=0, help='the id of the task to plot')
    return parser.parse_args()

def gen_source(args, expid=None):
    my_source = f'./data/KRR_{args.dataset}_ntask{args.ntask}_bandit{args.bmin}-{args.bmax}-{args.eta}_Nloop{args.Nloop}_expid{expid}.pkl'
    return my_source


def data_process_single_exp(args, expid=None):
    my_source = gen_source(args, expid=expid)
    print("source:", my_source)
    results_summary = pickle.load(open(my_source, "rb"))
    # print("results_summary")
    # print(results_summary)
    
    styles = {'GPTuneBand': ['GPTuneBand','#C33734','solid'],
              'GPTune': ['GPTune','#2ca02c', 'solid'], 
              'hpbandster': ['HpBandSter', '#ff7f0e', 'dashed'],
              'TPE': ['TPE', '#9467bd', 'dashdot'],
              'opentuner': ['OpenTuner', '#1f77b4',  'dotted']}

    tuners = []
    for item in results_summary:
        tuners.append(item[0])
    
    data_summary_dict = {}
    for j, item in enumerate(results_summary):
        tuner_data_summary = {}
        for i in range(args.ntask):
            # tuning_history_data = historical_best(item[i+1][2][1])
            tuning_history_data = [-x/100 for x in historical_best(item[i+1][2][1])]
            tuner_data_summary[item[i+1][1][5:]] = [item[i+1][2][0], tuning_history_data]
        data_summary_dict[tuners[j]] = tuner_data_summary
        print("Processed data")
        print(tuner_data_summary)
        print()
    return data_summary_dict, styles

def data_process(args):
    # if len(args.explist) == 1:
    #     return data_process_single_exp(args, expid=args.explist[0])
    # else:
    data_summary_dict_set = []
    for expid in args.explist:
        data_summary_dict_cur, styles = data_process_single_exp(args, expid=expid)
        data_summary_dict_set.append(data_summary_dict_cur)
    return data_summary_dict_set, styles
    
def historical_best(data):
    for i in range(len(data)-1):
        data[i+1] = data[i] if data[i] < data[i+1] else data[i+1]
    return data


def plot_history(args, data_summary_dict_set, styles, baseline=None,
                 deleted_tuners=None):

    N_tuners = len(data_summary_dict_set[0].keys())
    tuners = list(data_summary_dict_set[0].keys())
    N_exps = len(args.explist)
    assert N_exps == len(data_summary_dict_set)
    
    # get a list of tasks
    values_view = data_summary_dict_set[0].values()
    value_iterator = iter(values_view)
    first_tuner_info = next(value_iterator)
    task_set = list(first_tuner_info.keys())

    for task in task_set:
        print(task)
        bugets_set = []
        mean_results_set = []
        std_results_set = []
        for i,tuner in enumerate(tuners):
            tuner_info = data_summary_dict_set[0][tuner]
            bugets_set.append(data_summary_dict_set[0][tuner][task][0])
            results_set=[]
            for exp_j in range(N_exps):
                results_set.append(data_summary_dict_set[exp_j][tuner][task][1])
            # print("results set:")
            mean_results_set.append(np.mean(results_set, axis=0))
            std_results_set.append(np.std(results_set, ddof=0,axis=0))
            if tuner == "GPTuneBand":
                print(tuner)
                print("Mean and std")
                print(bugets_set[i])
                print(mean_results_set[i])
                print(std_results_set[i])
                print()
        
            
        expname = '-'.join(args.explist)
        my_source = gen_source(args, expid=expname)
        filename = os.path.splitext(os.path.basename(my_source))[0]
            
        fig, ax = plt.subplots(nrows=1, ncols=1)
        if baseline != None:
            ax.plot(bugets_set[0], 
                    args.fmin*np.ones(len(bugets_set)), 
                    'k--', label="True")

        for i in range(N_tuners):
            if args.deleted_tuners == None or tuners[i] not in args.deleted_tuners:
                ax.plot(bugets_set[i], 
                        mean_results_set[i],
                        color=styles[tuners[i]][1], 
                        label=styles[tuners[i]][0],
                        linestyle=styles[tuners[i]][2])
                ax.fill_between(bugets_set[i], 
                                mean_results_set[i]+std_results_set[i], 
                                mean_results_set[i]-std_results_set[i],
                                color=styles[tuners[i]][1], 
                                alpha=0.3)
        ax.legend(fontsize=8)
        savepath = os.path.join("./plots", f"Tuning_history_{filename}_{task}.pdf")    
        plt.savefig(savepath)
        print("Figure saved: ", savepath)

def main(args):
    print()
    print("plotting args:", args)
    data_summary_dict_set, styles = data_process(args)
    plot_history(args, data_summary_dict_set, styles)

    
if __name__ == "__main__":
    main(parse_args())