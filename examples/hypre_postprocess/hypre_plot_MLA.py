import os
import os.path as osp
import argparse
import pickle 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nmax', type=int, default=100, help='maximum discretization size')
    parser.add_argument('--nmin', type=int, default=10, help='minimum discretization size')
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument("--equation", type=str, default="poisson", help ='type of PDE to solve')
    parser.add_argument('--nrun', type=int, default=10, help='number of runs')
    parser.add_argument('--multistart', type=int, default=None, help='number of model restarts')
    parser.add_argument('--bandit', type=int, default=None, help='use bandit strategy in HpBandster or not')
    parser.add_argument('--tuner4', type=int, default=0, help='4 tuners or the usual 3 tuners')
    parser.add_argument('--rerun', type=int, default=0, help='rerun id')
    parser.add_argument('--average', type=int, default=0, help='number of history to average on')
    return parser.parse_args()

def gen_source(args):
    if args.tuner4 == True and args.multistart > 0:
        if args.rerun == 0:
            my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}_tuner4.pkl"
        else:
            my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}_tuner4_rerun{args.rerun}.pkl"
    elif args.multistart == None and args.bandit == None:
        my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}.pkl"
    elif args.bandit != None and args.multistart == None:
        my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_bandit.pkl"
    elif args.multistart > 0 and args.bandit == None:
        my_source = f"./data_MLA/exp_hypre_{args.equation}_nmax{args.nmax}_nmin{args.nmin}_ntask{args.ntask}_multistart{args.multistart}.pkl"
    else:
        raise NotImplementedError
    return my_source

def data_process(args, average_id = None):
    if average_id == None:
        my_source = gen_source(args)
        results_summary = pickle.load(open(my_source, "rb"))
    else:
        args.rerun = average_id
        my_source = gen_source(args)
        results_summary = pickle.load(open(my_source, "rb"))
    
    ntask = len(results_summary)
    assert ntask == args.ntask
    GPTune_time = []
    Opentuner_time = []
    HpBandster_time = []
    GPTune_multistart_time = []
    size_set = []
    for i in range(ntask):
        task_current = results_summary[i]
        idx_nrun = task_current[2].index(args.nrun)
        results_GPTune = task_current[3][0]
        results_OpenTuner = task_current[3][1]
        results_HpBandster = task_current[3][2]
        assert results_GPTune[0] == "GPTune"
        assert results_OpenTuner[0] == "OpenTuner"
        assert results_HpBandster[0] == "Hpbandster"
        # OpenTunervsGPTune.append(results_OpenTuner[1][idx_nrun]/results_GPTune[1][idx_nrun])
        # HpBandstervsGPTune.append(results_HpBandster[1][idx_nrun]/results_GPTune[1][idx_nrun])
        GPTune_time.append(results_GPTune[1][idx_nrun])
        Opentuner_time.append(results_OpenTuner[1][idx_nrun])
        HpBandster_time.append(results_HpBandster[1][idx_nrun])
        size_set.append(task_current[4])
        if args.tuner4:
            results_GPTune_multistart = task_current[3][3]
            assert results_GPTune_multistart[0] == "GPTune_multistart5"
            GPTune_multistart_time.append(results_GPTune_multistart[1][idx_nrun])
    time_summary = [np.array(GPTune_time), np.array(Opentuner_time), np.array(HpBandster_time), np.array(GPTune_multistart_time)]
    return time_summary, size_set

def plot(GPTune_time, Opentuner_time, HpBandster_time, args, GPTune_multistart_time = []):
    my_source = gen_source(args)
    filename = os.path.splitext(os.path.basename(my_source))[0]
    nrun = args.nrun
    if len(GPTune_multistart_time) == 0:
        savepath = os.path.join("./plots_MLA", f"{filename}_nrun{nrun}_average{args.average}.pdf")
        data1 = Opentuner_time/GPTune_time
        data2 = HpBandster_time/GPTune_time
    else:
        savepath = os.path.join("./plots_MLA", f"{filename}_nrun{nrun}_average{args.average}_GPTunemultistart.pdf")
        data1 = Opentuner_time/GPTune_multistart_time
        data2 = HpBandster_time/GPTune_multistart_time
    
    assert len(data1) == len(data2) 
    ntask = len(data1)
    
    p1 = len([x for x in data1 if x >= 1])
    p2 = len([x for x in data2 if x >= 1])
    p3 =  len([x for x in data1 if x < 0.5])
    p4 =  len([x for x in data2 if x < 0.5])
    
    # plot    
    plt.clf()
    x = np.arange(1, ntask+1)
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    # ax.bar(x - width/2, data1, width, label=f'OpenTuner/GPTune, {p1}(>=1), {p3}(<0.5)')
    # ax.bar(x + width/2, data2, width, label=f'HpBandster/GPTune, {p2}(>=1), {p4}(<0.5)')
    if args.bandit == None:
        ax.bar(x - width/2, data1, width, label=f'OpenTuner/GPTune', color='#1f77b4')
        ax.bar(x + width/2, data2, width, label=f'HpBandster/GPTune', color='#ff7f0e')
    else:
        ax.bar(x - width/2, data1, width, label=f'OpenTuner/GPTune', color='#1f77b4')
        ax.bar(x + width/2, data2, width, label=f'HpBandster_bandit/GPTune', color='#ff7f0e')
        
    ax.plot([0,ntask+1], [1, 1], c='black', linestyle=':')
    ax.plot([0,ntask+1], [0.5, 0.5], linestyle=':', linewidth=1)
    ax.set_ylabel('Ratio of best performance')
    ax.set_xlabel('Task ID')
    # equation_name = "Poisson"
    # if args.equation == "convdiff":
    #     equation_name = "Convection-diffusion"
    # ax.set_title(f'{equation_name}, [nx, ny, nz] in [{args.nmin}, {args.nmax}], nrun = {nrun}')   
    ax.legend(fontsize=8)
    fig.tight_layout()
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

def plot_histogram(GPTune_time, Opentuner_time, HpBandster_time, args, GPTune_multistart_time = []):
    my_source = gen_source(args)
    filename = os.path.splitext(os.path.basename(my_source))[0]
    nrun = args.nrun
    if len(GPTune_multistart_time) == 0:
        savepath = os.path.join("./plots_MLA", f"hist_{filename}_nrun{nrun}_average{args.average}.pdf")
        data1 = Opentuner_time/GPTune_time
        data2 = HpBandster_time/GPTune_time
    else:
        print("GPTune with multistart")
        savepath = os.path.join("./plots_MLA", f"hist_{filename}_nrun{nrun}_average{args.average}_GPTune_multistart.pdf")
        data1 = Opentuner_time/GPTune_multistart_time
        data2 = HpBandster_time/GPTune_multistart_time
    assert len(data1) == len(data2) 
    ntask = len(data1)
    
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    rects1 = ax1.hist(data1, bins=np.arange(0, int(np.ceil(max(data1)))+1, 0.25), weights=np.ones(ntask) / ntask, 
             label=f'OpenTuner/GPTune', color='#1f77b4')
    if args.bandit == None:
        rects2 = ax2.hist(data2, bins=np.arange(0, int(np.ceil(max(data2)))+1, 0.25) , weights=np.ones(ntask) / ntask, 
                label=f'HpBandster/GPTune', color='#ff7f0e')
    else:
        assert GPTune_multistart_time == None, "Can't deal with HpBandster_bandit and GPTune_multistart for now."
        rects2 = ax2.hist(data2, bins=np.arange(0, int(np.ceil(max(data2)))+1, 0.25) , weights=np.ones(ntask) / ntask, 
                label=f'HpBandster_bandit/GPTune', color='#ff7f0e')
        
    ax1.plot([1, 1], [0, 1], c='black', linestyle=':')
    ax2.plot([1, 1], [0, 1], c='black', linestyle=':')
    ax1.set_xticks(np.arange(0, int(np.ceil(max(data1)))+1, step=0.5))
    ax2.set_xticks(np.arange(0, int(np.ceil(max(data2)))+1, step=0.5))
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    autolabel(rects1, ax1)
    autolabel(rects2, ax2)

    # equation_name = "Poisson"
    # if args.equation == "convdiff":
    #     equation_name = "Convection-diffusion"
    # ax1.set_title(f'{equation_name}, [nx, ny, nz] in [{args.nmin}, {args.nmax}], nrun = {nrun}')
    ax1.set_ylabel('Density')
    ax2.set_ylabel('Density')
    ax2.set_xlabel('Ratio of best performance')
    fig.tight_layout()
    fig.savefig(savepath)
    print("Figure saved:, ", savepath)

    
def plot_size_time(data1, data2, data3, size_set, args):
    my_source = gen_source(args)
    assert len(data1) == len(data2)
    assert len(data1) == len(data3)
    nrun = args.nrun
    size_set = np.array(size_set)
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    
    # fit a linear line
    size_set = size_set.reshape(-1, 1)
    reg1 = LinearRegression().fit(np.log(size_set), np.log(data1))
    reg2 = LinearRegression().fit(np.log(size_set), np.log(data2))
    reg3 = LinearRegression().fit(np.log(size_set), np.log(data3))
    
    # plot
    plt.clf()
    plt.loglog(size_set, data1, label=f'GPTune, coeff={reg1.coef_[0]:.2f}', color='#2ca02c')
    plt.loglog(size_set, data2, label=f'OpenTuner, coeff={reg2.coef_[0]:.2f}', color='#1f77b4')
    plt.loglog(size_set, data3, label=f'HpBandster, coeff={reg3.coef_[0]:.2f}', color='#ff7f0e')
    # plt.loglog(size_set, np.array(size_set) + (data1[0]+data2[0]+data3[0])/3-size_set[0], linestyle='--', color='red')
    plt.xlabel('Problem size nx*ny*nz')
    plt.ylabel('Optimal Hypre Time')
    plt.legend(fontsize=8)
    plt.grid()
    # equation_name = "Poisson"
    # if args.equation == "convdiff":
    #     equation_name = "Convection-diffusion"
    # plt.title(f'{equation_name}, [nx, ny, nz] in [{args.nmin}, {args.nmax}], nrun = {nrun}')
    filename = os.path.splitext(os.path.basename(my_source))[0]
    plt.savefig(os.path.join("./plots_MLA", f"sizetime_{filename}_nrun{nrun}.pdf"))
    
    
def main(args):
    if args.average == 0:
        time_summary, size_set = data_process(args)
        GPTune_time = time_summary[0]
        Opentuner_time = time_summary[1]
        HpBandster_time = time_summary[2]
        GPTune_multistart_time = time_summary[3] if args.tuner4 else []
    else:
        GPTune_time = []
        Opentuner_time = []
        HpBandster_time = []
        GPTune_multistart_time = []
        for i in range(args.average+1):
            print(f"Averaging ... , i = {i}")
            time_summary, size_set = data_process(args, average_id = i)
            GPTune_time.append(time_summary[0])
            Opentuner_time.append(time_summary[1])
            HpBandster_time.append(time_summary[2])
            GPTune_multistart_time.append(time_summary[3])
        print("GPTune_time, ", GPTune_time)
        print("Opentuner_time, ", Opentuner_time)
        print("HpBandster_time, ", HpBandster_time)
        print("GPTune_multistart_time, ", GPTune_multistart_time)
        GPTune_time.pop(2)
        Opentuner_time.pop(2)
        HpBandster_time.pop(2)
        GPTune_time = np.mean(GPTune_time, axis=0)
        Opentuner_time = np.mean(Opentuner_time, axis=0)
        HpBandster_time = np.mean(HpBandster_time, axis=0)
        GPTune_multistart_time = np.mean(GPTune_multistart_time, axis=0)
        print("Average GPTune_time, ", GPTune_time)
        print("Average Opentuner_time, ", Opentuner_time)
        print("Average HpBandster_time, ", HpBandster_time)
        print("Average GPTune_multistart_time, ", GPTune_multistart_time)

    # plot(GPTune_time, Opentuner_time, HpBandster_time, args, GPTune_multistart_time = GPTune_multistart_time)
    # plot(GPTune_time, Opentuner_time, HpBandster_time, args, GPTune_multistart_time = [])
    plot_histogram(GPTune_time, Opentuner_time, HpBandster_time, args, GPTune_multistart_time = GPTune_multistart_time)
    plot_histogram(GPTune_time, Opentuner_time, HpBandster_time, args, GPTune_multistart_time = [])
    # plot_size_time(GPTune_time, Opentuner_time, HpBandster_time, size_set, args)
    
if __name__ == "__main__":
    main(parse_args())