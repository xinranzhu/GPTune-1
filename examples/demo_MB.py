#! /usr/bin/env python

# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#


################################################################################

from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from gptune import GPTune_MB
from data import Data
from data import Categoricalnorm
from options import Options
from computer import Computer
import sys
import os
import mpi4py
from mpi4py import MPI
import numpy as np
import time
import argparse
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import callhpbandster_bandit
import logging

sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))
logging.getLogger('matplotlib.font_manager').disabled = True

# from GPTune import *

################################################################################

# Define Problem

# YL: for the spaces, the following datatypes are supported:
# Real(lower, upper, transform="normalize", name="yourname")
# Integer(lower, upper, transform="normalize", name="yourname")
# Categoricalnorm(categories, transform="onehot", name="yourname")


# Argmin{x} objectives(t,x), for x in [0., 1.]

# bandit structure
bmin = 1
bmax = 27
eta = 3

def objectives(point):
    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    t = point['t']
    x = point['x']
    if 'budget' in point:
        bgt = point['budget']    
    else:
        bgt = bmax

    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1

    # print('test:',test)
    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)
    def perturb(bgt):
        perturb_magnitude = 0.1
        k1 = -perturb_magnitude/bmax
        # return np.cos(c)*(-np.log10(bgt))*0.1
        assert k1*bmax + perturb_magnitude == 0
        return np.cos(c) * (k1*bgt + perturb_magnitude)
    
    out = [f*(1+perturb(bgt))]
    print(f"One demo run, x = {x:.4f}, t = {t:.4f}, budget = {bgt:.4f}, perturb = {perturb(bgt):.4f}, out = {out[0]:.4f}")
    return out

def models(point):
    """
    f(t,x) = exp(- (x + 1) ^ (t + 1) * cos(2 * pi * x)) * (sin( (t + 2) * (2 * pi * x) ) + sin( (t + 2)^(2) * (2 * pi * x) + sin ( (t + 2)^(3) * (2 * pi *x))))
    """
    # global test
    t = point['t']
    x = point['x']
    a = 2 * np.pi
    b = a * t
    c = a * x
    d = np.exp(- (x + 1) ** (t + 1)) * np.cos(c)
    e = np.sin((t + 2) * c) + np.sin((t + 2)**2 * c) + np.sin((t + 2)**3 * c)
    f = d * e + 1
    # print('dd',test)

    """
    f(t,x) = x^2+t
    """
    # t = point['t']
    # x = point['x']
    # f = 20*x**2+t
    # time.sleep(1.0)

    return [f*(1+np.random.uniform()*0.1)]


""" Plot the objective function for t=1,2,3,4,5,6 """
def annot_min(x,y, ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    text= "x={:.3f}, y={:.3f}".format(xmin, ymin)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="-",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="offset points",arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(210,5), **kw)


def main():
    
    import matplotlib.pyplot as plt
    
    args = parse_args()
    ntask = args.ntask
    nodes = args.nodes
    cores = args.cores
    Nloop = args.Nloop
    TUNER_NAME = args.optimization
    
    os.environ['TUNER_NAME'] = TUNER_NAME

    input_space = Space([Real(0., 10., transform="normalize", name="t")])
    parameter_space = Space([Real(0., 1., transform="normalize", name="x")])
    # input_space = Space([Real(0., 0.0001, "uniform", "normalize", name="t")])
    # parameter_space = Space([Real(-1., 1., "uniform", "normalize", name="x")])

    output_space = Space([Real(float('-Inf'), float('Inf'), name="time")])
    constraints = {"cst1": "x >= 0. and x <= 1."}
    problem = TuningProblem(input_space, parameter_space,output_space, objectives, constraints, models)  # with performance model
    # problem = TuningProblem(input_space, parameter_space, output_space, objectives, constraints, None)  # no performance model
    computer = Computer(nodes=1, cores=cores, hosts=None)
    options = Options()
    options['model_restarts'] = 1
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    options['objective_evaluation_parallelism'] = False
    options['objective_multisample_threads'] = 1
    options['objective_multisample_processes'] = 1
    options['objective_nprocmax'] = 1
    options['model_processes'] = 1
    # options['model_threads'] = 1
    # options['model_restart_processes'] = 1
    # options['search_multitask_processes'] = 1
    # options['search_multitask_threads'] = 1
    # options['search_threads'] = 16
    # options['mpi_comm'] = None
    # options['mpi_comm'] = mpi4py.MPI.COMM_WORLD
    options['model_class'] = 'Model_LCM' if args.LCMmodel == 'LCM' else 'Model_GPy_LCM' # Model_GPy_LCM or Model_LCM
    options['verbose'] = True
    options['sample_class'] = 'SampleLHSMDU'
    options['sample_algo'] = 'LHS-MDU'
    options.validate(computer=computer)

    options['budget_min'] = bmin
    options['budget_max'] = bmax
    options['budget_base'] = eta
    smax = int(np.floor(np.log(options['budget_max']/options['budget_min'])/np.log(options['budget_base'])))
    budgets = [options['budget_max'] /options['budget_base']**x for x in range(smax+1)]
    NSs = [int((smax+1)/(s+1))*options['budget_base']**s for s in range(smax+1)] 
    NSs_all = NSs.copy()
    budget_all = budgets.copy()
    for s in range(smax+1):
        for n in range(s):
            NSs_all.append(int(NSs[s]/options['budget_base']**(n+1)))
            budget_all.append(int(budgets[s]*options['budget_base']**(n+1)))
    Ntotal = int(sum(NSs_all) * Nloop)
    Btotal = int(np.dot(np.array(NSs_all), np.array(budget_all))/options['budget_max']) # total number of evaluations at highest budget -- used for single-fidelity tuners
    print("samples in one multi-armed bandit loop, NSs_all = ", NSs_all)
    print("total number of samples: ", Ntotal)
    print("total number of evaluations at highest budget: ", Btotal)
    print(f"Sampler: {options['sample_class']}, {options['sample_algo']}")
    print()
    
    data = Data(problem)
    # giventask = [[1.0], [5.0], [10.0]]
    # giventask = [[1.0], [1.2], [1.3]]
    giventask = [[1.0], [1.2]]
    # t_end = args.t_end
    # giventask = [[i] for i in np.arange(1, t_end, (t_end-1)/ntask).tolist()] # 10 tasks
    # giventask = [[i] for i in np.arange(1.0, 6.0, 1.0).tolist()] # 5 tasks
    NI=len(giventask)
    assert NI == ntask # make sure number of tasks match
	    
    np.set_printoptions(suppress=False, precision=4)
    if(TUNER_NAME=='GPTuneBand'):
        NS = Nloop
        data = Data(problem)
        gt = GPTune_MB(problem, computer=computer, NS=Nloop, options=options)
        (data, stats, data_hist)=gt.MB_LCM(NS = Nloop, Igiven = giventask)
        print("Tuner: ", TUNER_NAME)
        print("Sampler class: ", options['sample_class'])
        print("Model class: ", options['model_class'])
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [a_val, c_val] = [{data.I[tid][0]:.3f}, {data.I[tid][1]:.3f}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            nth = np.argmin(data.O[tid])
            Popt = data.P[tid][nth]
            # find which arm and which sample the optimal param is from
            for arm in range(len(data_hist.P)):
                try:
                    idx = (data_hist.P[arm]).index(Popt)
                    arm_opt = arm
                except ValueError:
                    pass
            print('    Popt ', Popt, 'Oopt ', min(data.O[tid])[0], 'nth ', nth, 'nth-bandit (s, nth) = ', (arm_opt, idx))
         
    if(TUNER_NAME=='GPTune'):
        NS = Btotal
        if args.nrun > 0:
            NS = args.nrun
        NS1 = max(NS//2, 1)
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=NS1)
        print("stats: ", stats)
        print("Sampler class: ", options['sample_class'], "Sample algo:", options['sample_algo'])
        print("Model class: ", options['model_class'])
        if options['model_class'] == 'Model_LCM' and NI > 1:
            print("Get correlation metric ... ")
            C = model[0].M.kern.get_correlation_metric()
            print("The correlation matrix C is \n", C)
        elif options['model_class'] == 'Model_GPy_LCM' and NI > 1:
            print("Get correlation metric ... ")
            C = model[0].get_correlation_metric(NI)
            print("The correlation matrix C is \n", C)

        
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], f'Oopt  {min(data.O[tid])[0]:.3f}', 'nth ', np.argmin(data.O[tid]))
            
    if(TUNER_NAME=='opentuner'):
        NS = Btotal
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid][:NS])], 'Oopt ', min(data.O[tid][:NS])[0], 'nth ', np.argmin(data.O[tid][:NS]))
            
    if(TUNER_NAME=='hpbandster'):
        NS = Btotal
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            
    
    if(TUNER_NAME=='TPE'):
        NS = Ntotal
        (data,stats)=callhpbandster_bandit.HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, options=options, run_id="hpbandster_bandit", niter=1)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"    t: {data.I[tid][0]:.2f} ")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            # print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
            max_budget = 0.
            Oopt = 99999
            Popt = None
            nth = None
            for idx, (config, out) in enumerate(zip(data.P[tid], data.O[tid].tolist())):
                for subout in out[0]:
                    budget_cur = subout[0]
                    if budget_cur > max_budget:
                        max_budget = budget_cur
                        Oopt = subout[1]
                        Popt = config
                        nth = idx
                    elif budget_cur == max_budget:
                        if subout[1] < Oopt:
                            Oopt = subout[1]
                            Popt = config
                            nth = idx                    
            print('    Popt ', Popt, 'Oopt ', Oopt, 'nth ', nth)


    plot=0
    if plot==1:
        x = np.arange(0., 1., 0.00001)
        Nplot=1.5
        for t in np.linspace(0,Nplot,2):
            fig = plt.figure(figsize=[12.8, 9.6])
            I_orig=[t]
            kwargst = {input_space[k].name: I_orig[k] for k in range(len(input_space))}

            y=np.zeros([len(x),1])
            for i in range(len(x)):
                P_orig=[x[i]]
                kwargs = {parameter_space[k].name: P_orig[k] for k in range(len(parameter_space))}
                kwargs.update(kwargst)
                y[i]=objectives(kwargs) 
            fontsize=30
            plt.rcParams.update({'font.size': 21})
            plt.plot(x, y, 'b')
            plt.xlabel('x',fontsize=fontsize+2)
            plt.ylabel('y(t,x)',fontsize=fontsize+2)
            plt.title('t=%d'%t,fontsize=fontsize+2)
            print('t:',t,'x:',x[np.argmin(y)],'ymin:',y.min())    
        
            annot_min(x,y)
            # plt.show()
            # plt.show(block=False)
            fig.savefig('obj_t_%d.eps'%t)                

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=1, help='Number of tasks')
    parser.add_argument('-t_end', type=float, default=2.0, help='end of task value')
    parser.add_argument('-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cpu cores')
    parser.add_argument('-nrun', type=int, default=-1, help='total application runs')
    parser.add_argument('-LCMmodel', type=str, default='LCM', help='choose from LCM models: LCM or GPy_LCM')
    parser.add_argument('-Nloop', type=int, default=1, help='Number of outer loops in multi-armed bandit per task')
    # parser.add_argument('-sample_class', type=str,default='SampleOpenTURNS',help='Supported sample classes: SampleLHSMDU, SampleOpenTURNS')
    args = parser.parse_args()
    
    return args   


if __name__ == "__main__":
    main()