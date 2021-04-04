import os
import os.path as osp
import argparse
import pickle
from operator import itemgetter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntask', type=int, default=20, help='number of tasks')
    parser.add_argument('--eta', type=int, default=3, help='eta value in bandit structure')
    parser.add_argument('--bmax', type=int, default=100, help='maximum budget in bandit structure')
    parser.add_argument('--bmin', type=int, default=10, help='minimum budget in bandit structure')
    parser.add_argument('--expid', type=str, default=None, help='experiment id')
    return parser.parse_args()

def main(args):
    my_source = f"./data/data_demo_ntask{args.ntask}_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}_expid{args.expid}.txt"
    save_path = f"./data/data_demo_ntask{args.ntask}_bmin{args.bmin}_bmax{args.bmax}_eta{args.eta}_expid{args.expid}.pkl"

    summary = []
    tasks = []
    data1 = ["GPTune_MB"]
    data2 = ["HpBandster"]
    data3 = ["GPTune"]
    data4 = ["OpenTuner"]
    data5 = ["TPE"]
    mean1 = []
    mean2 = []
    mean3 = []
    mean4 = []
    mean5 = []
    std1 = []
    std2 = []
    std3 = []
    std4 = []
    std5 = []
    hist1 = []
    hist2 = []
    hist3 = []
    hist4 = []
    hist5 = []

    with open(my_source, "r") as f:
        line = f.readline()
        while line:
            info = line.split()
            if info[0] == "GPTuneBand":
                taskid = int(info[1][:-1])
                t_val = float(info[4])
                tasks.append([taskid, t_val])
                mean1.append(float(info[5]))
                std1.append(float(info[6]))
                hist_task = []
                for i in range(7, len(info)):
                    hist_task.append(float(info[i]))
                hist1.append(hist_task)
            elif info[0] == "HpBandster":
                mean2.append(float(info[1]))
                std2.append(float(info[2]))
                hist_task = []
                for i in range(3, len(info)):
                    hist_task.append(float(info[i]))
                hist2.append(hist_task)
            elif info[0] == "GPTune":
                mean3.append(float(info[1]))
                std3.append(float(info[2]))
                hist_task = []
                for i in range(3, len(info)):
                    hist_task.append(float(info[i]))
                hist3.append(hist_task)
            elif info[0] == "OpenTuner":
                mean4.append(float(info[1]))
                std4.append(float(info[2]))
                hist_task = []
                for i in range(3, len(info)):
                    hist_task.append(float(info[i]))
                hist4.append(hist_task)
            elif info[0] == "TPE":
                mean5.append(float(info[1]))
                std5.append(float(info[2]))
                hist_task = []
                for i in range(3, len(info)):
                    hist_task.append(float(info[i]))
                hist5.append(hist_task)
            else:
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
    data4 = my_append(data4, mean4, std4, hist4)
    data5 = my_append(data5, mean5, std5, hist5)
    summary = [tasks, data1, data2, data3, data4, data5]
    pickle.dump(summary, open(save_path, "wb"))  

    print("data1 = ", data1)
    print("data2 = ", data2)
    print("data3 = ", data3)
    print("data4 = ", data4)
    print("data5 = ", data5)
    print("summary = ", summary)
          
if __name__ == "__main__":
    main(parse_args())
