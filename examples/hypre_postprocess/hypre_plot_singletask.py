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
    parser.add_argument('--ntask', type=int, default=30, help='number of tasks')
    parser.add_argument("--equation", type=str, default="Poisson", help ='type of PDE to solve')
    parser.add_argument('--nrun', type=int, default=10, help='number of runs')
    # parser.add_argument("--typeofresults", type=str, default="performance", help ='type of results')
    return parser.parse_args()

def gen_source(args):
    my_source = [f"./data_singletask/exp_hypre_{args.equation}_singletask{args.ntask}_performance.pkl", f"./exp_hypre_{args.equation}_singletask{args.ntask}_time.pkl"]
    return my_source

def data_process(args):
    my_source = gen_source(args)
    results_perform = pickle.load(open(my_source[0], "rb"))
    results_time = pickle.load(open(my_source[1], "rb"))
    ntask = len(results_perform)
    assert ntask == args.ntask
    assert ntask == len(results_time)
    HpBandstervsGPTune_perform = []
    HpBandstervsGPTune_time = []
    GPTune_perform = []
    HpBandster_perform = []
    GPTune_time = []
    HpBandster_time = []
    size_set = []
    for i in range(ntask):
        # performance record
        task_current = results_perform[i]
        idx_nrun = task_current[2].index(args.nrun)
        results_GPTune = task_current[3][0]
        results_HpBandster = task_current[3][2]
        assert results_GPTune[0] == "GPTune"
        assert results_HpBandster[0] == "Hpbandster"
        HpBandstervsGPTune_perform.append(results_HpBandster[1][idx_nrun]/results_GPTune[1][idx_nrun])
        GPTune_perform.append(results_GPTune[1][idx_nrun])
        HpBandster_perform.append(results_HpBandster[1][idx_nrun])
        size_set.append(task_current[4])
        
        # tuner runtime record
        task_current_time = results_time[i]
        results_GPTune_time = task_current_time[3][0]
        results_HpBandster_time = task_current_time[3][2]
        assert results_GPTune_time[0] == "GPTune"
        assert results_HpBandster_time[0] == "Hpbandster"
        HpBandstervsGPTune_time.append(results_HpBandster_time[1][idx_nrun]/results_GPTune_time[1][idx_nrun])
        GPTune_time.append(results_GPTune_time[1][idx_nrun])
        HpBandster_time.append(results_HpBandster_time[1][idx_nrun])
    
    summary_perform = [HpBandstervsGPTune_perform, GPTune_perform, HpBandster_perform]
    summary_time = [HpBandstervsGPTune_time, GPTune_time, HpBandster_time]
    return  summary_perform, summary_time, size_set

def plot(data2, args):
    my_source = gen_source(args)
    nrun = args.nrun
    ntask = args.ntask
    assert len(data2) == ntask
    
    p2 = len([x for x in data2 if x >= 1])
    p4 =  len([x for x in data2 if x < 0.5])
    
    # plot    
    plt.clf()
    x = np.arange(1, ntask+1)
    width = 0.35  # the width of the bars
    plt.bar(x + width/2, data2, width, label=f'HpBandster/GPTune', color='#ff7f0e')
    plt.plot([0,ntask+1], [1, 1], c='black', linestyle=':')
    plt.plot([0,ntask+1], [0.5, 0.5], linestyle=':', linewidth=1)
    plt.ylabel('Ratio of best performance')
    plt.xlabel('Task ID') 
    plt.legend(fontsize=8)
    plt.tight_layout()
    filename = os.path.splitext(os.path.basename(my_source[0]))[0]
    plt.savefig(os.path.join("./plots_singletask", f"{filename}_nrun{nrun}.pdf"))

def plot_histogram(data2, args):
    my_source = gen_source(args)
    nrun = args.nrun
    ntask = args.ntask
    assert len(data2) == ntask
    
    plt.clf()
    rects1 = plt.hist(data2, bins=np.arange(0, int(np.ceil(max(data2)))+1, 0.5) , weights=np.ones(ntask) / ntask, 
             label=f'HpBandster/GPTune, range: [{min(data2):.2f}, {max(data2):.2f}]', color='#ff7f0e')
    plt.plot([1, 1], [0, 1], c='black', linestyle=':')
    plt.plot([1, 1], [0, 1], c='black', linestyle=':')
    plt.legend(fontsize=8)
    plt.ylabel('Fraction')
    plt.xlabel('Ratio of best performance')
    
    def autolabel(rects):
    # """Attach a text label above each bar in *rects*, displaying its height."""
        for i in range(len(rects[0])):
            cur_height = rects[0][i]
            if cur_height != 0:
                plt.annotate(f'{cur_height:.2f}',
                            xy=((rects[1][i] + rects[1][i+1])/2, cur_height),
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1)

    # fig.tight_layout()
    filename = os.path.splitext(os.path.basename(my_source[0]))[0]
    plt.savefig(os.path.join("./plots_singletask", f"hist_{filename}_nrun{nrun}.pdf"))
    
    
def scatter_2d(data_perform, data_time, args):
    my_source = gen_source(args)
    nrun = args.nrun
    ntask = args.ntask
    assert len(data_perform) == ntask
    assert len(data_time) == ntask
    
    plt.clf()
    plt.scatter(data_perform, data_time, label=f'HpBandster/GPTune', color='#ff7f0e')
    filename = os.path.splitext(os.path.basename(my_source[0]))[0]
    plt.plot([1, 1], [0, 3], c='black', linestyle=':')
    plt.plot([0, 3], [1, 1], c='black', linestyle=':')
    plt.ylabel('Ratio total runtime')
    plt.xlabel('Ratio best performance')
    plt.legend(fontsize=8)
    plt.savefig(os.path.join("./plots_singletask", f"scatter_{filename}_nrun{nrun}.pdf"))

    


def plot_size_time(data1, data3, size_set, args):
    my_source = gen_source(args)
    assert len(data1) == len(data3)
    nrun = args.nrun
    size_set = np.array(size_set)
    data1 = np.array(data1)
    data3 = np.array(data3)
    
    # fit a linear line
    size_set = size_set.reshape(-1, 1)
    reg1 = LinearRegression().fit(np.log(size_set), np.log(data1))
    reg3 = LinearRegression().fit(np.log(size_set), np.log(data3))
    
    # plot
    plt.clf()
    plt.loglog(size_set, data1, label=f'GPTune, coeff={reg1.coef_[0]:.2f}', color='#2ca02c')
    plt.loglog(size_set, data3, label=f'HpBandster, coeff={reg3.coef_[0]:.2f}', color='#ff7f0e')
    plt.xlabel('Problem size nx*ny*nz')
    plt.ylabel('Optimal Hypre Time')
    plt.legend(fontsize=8)
    plt.grid()
    # equation_name = "Poisson"
    # if args.equation == "convdiff":
    #     equation_name = "Convection-diffusion"
    # plt.title(f'{equation_name}, [nx, ny, nz] in [{args.nmin}, {args.nmax}], nrun = {nrun}')
    filename = os.path.splitext(os.path.basename(my_source[0]))[0]
    plt.savefig(os.path.join("./plots_singletask", f"sizetime_{filename}_nrun{nrun}.pdf"))
    
  
def main(args):
    summary_perform, summary_time, size_set = data_process(args)
    HpBandstervsGPTune_perform = summary_perform[0]
    GPTune_perform = summary_perform[1]
    HpBandster_perform = summary_perform[2]
    HpBandstervsGPTune_time = summary_time[0]
    
    # plot(HpBandstervsGPTune_perform, args)
    # plot_histogram(HpBandstervsGPTune_perform, args)
    plot_size_time(GPTune_perform, HpBandster_perform, size_set, args)
    # scatter_2d(HpBandstervsGPTune_perform, HpBandstervsGPTune_time, args)
        
    
if __name__ == "__main__":
    main(parse_args())