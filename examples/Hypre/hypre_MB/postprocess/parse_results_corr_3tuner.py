import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bmax', type=int, default=100, help='maximum budget in bandit structure')
    parser.add_argument('--bmin', type=int, default=10, help='minimum budget in bandit structure')
    parser.add_argument('--eta', type=int, default=3, help='eta value in bandit structure')
    parser.add_argument('--ntask', type=str, default=None, help='number of tasks')
    parser.add_argument('--expid', type=str, default=None, help='experiment id')
    return parser.parse_args()

def main(args):
    my_source = f"./data/corr_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}_ntask{args.ntask}_expid{args.expid}.txt"
    save_path = f"./data/corr_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}_ntask{args.ntask}_expid{args.expid}.pkl"

    summary = []
    tasks = []
    data1 = ["GPTuneBand"]
    data2 = ["HpBandster"]
    data3 = ["GPTune"]
    
    mean1 = []
    mean2 = []
    mean3 = []

    std1 = []
    std2 = []
    std3 = []

    hist1 = []
    hist2 = []
    hist3 = []
 
    
    # summary = [[[0, [1.764, 0.169]], [1, [0.041, 0.809]], ..., [19, [1.764, 0.169]]], 
    #             ["GPTuneBand", [mean0, mean1, ..., mean19], [std0, std1, ...], [[data1, data2], [data1, data2]], 
    #             ["HpBandster", [mean0, mean1, ..., mean19], [std0, std1, ...], [data1], [data2]],  
    #           ]
    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split()
            if info[0] == "GPTuneBand":
                taskid = int(info[1][:-1])
                a_val = float(info[2][1:-1])
                c_val = float(info[3][:-1])
                tasks.append([taskid, [a_val, c_val]]) 
                mean1.append(float(info[4]))
                std1.append(float(info[5]))
                hist_task = []
                for i in range(6, len(info)):
                    hist_task.append(float(info[i]))
                hist1.append(hist_task)
            elif info[0] == "HpBandster":
                try:
                    mean2.append(float(info[1]))
                    std2.append(float(info[2]))
                    hist_task = []
                    for i in range(3, len(info)):
                        hist_task.append(float(info[i]))
                    hist2.append(hist_task)
                except ValueError:
                    pass 
            elif info[0] == "GPTune":
                mean3.append(float(info[1]))
                std3.append(float(info[2]))
                hist_task = []
                for i in range(3, len(info)):
                    hist_task.append(float(info[i]))
                hist3.append(hist_task)
            elif info[0] == "OpenTuner" or info[0] == "TPE" or info[0] == "GPTuneBandS":
                pass
            else:
                print(info[0])
                raise NotImplementedError()
            line = f.readline()

    def my_append(data1, mean1, std1, hist1):
        data1.append(mean1)
        data1.append(std1)
        data1.append(hist1)
        return data1
    
    data1 = my_append(data1, mean1, std1, hist1)
    data2 = my_append(data2, mean2, std2, hist2)
    data3 = my_append(data3, mean3, std3, hist3)
    summary = [tasks, data1, data2, data3]
    pickle.dump(summary, open(save_path, "wb"))  

    print("data1 = ", data1)
    print("data2 = ", data2)
    print("data3 = ", data3)
    print("summary = ", summary)
                        
                    
if __name__ == "__main__":
    main(parse_args())
                    
            