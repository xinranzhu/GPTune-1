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
"""
Example of invocation of this script:

python hypre.py -nxmax 200 -nymax 200 -nzmax 200 -nxmin 100 -nymin 100 -nzmin 100 -nodes 1 -cores 32 -nprocmin_pernode 1 -ntask 20 -nrun 800 -machine cori -jobid 0

where:
    -nxmax/nymax/nzmax       maximum number of discretization size for each dimension
    -nodes                   number of compute node
    -cores                   number of cores per node
    -nprocmin_pernode is the minimum number of MPIs per node for launching the application code
    -machine                 name of the machine 
    -ntask                   number of different tasks to be tuned
    -nrun                    number of calls per task
    -jobid                   optional, can always be 0
    
Description of the parameters of Hypre AMG:
Task space:
    nx:    problem size in dimension x
    ny:    problem size in dimension y
    nz:    problem size in dimension z
    cx:    diffusion coefficient for d^2/dx^2
    cy:    diffusion coefficient for d^2/dy^2
    cz:    diffusion coefficient for d^2/dz^2
    ax:    convection coefficient for d/dx
    ay:    convection coefficient for d/dy
    az:    convection coefficient for d/dz
Input space:
    Px:                processor topology, with Nproc = Px*Py*Pz where Pz is a dependent parameter
    Py:                processor topology, with Nproc = Px*Py*Pz where Pz is a dependent parameter
    Nproc:             total number of MPIs 
    strong_threshold:  AMG strength threshold
    trunc_factor:      Truncation factor for interpolation
    P_max_elmts:       Max number of elements per row for AMG interpolation
    coarsen_type:      Defines which parallel coarsening algorithm is used
    relax_type:        Defines which smoother to be used
    smooth_type:       Enables the use of more complex smoothers
    smooth_num_levels: Number of levels for more complex smoothers
    interp_type:       Defines which parallel interpolation operator is used  
    agg_num_levels:    Number of levels of aggressive coarsening
"""
from hypredriver import hypredriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from gptune import GPTune_MB
from data import Data
from data import Categoricalnorm
from options import Options
from computer import Computer
import sys, os, re
import numpy as np
import time
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import callhpbandster_bandit
import math
import functools

# import mpi4py
# from mpi4py import MPI

# add GPTunde path in front of all python pkg path
sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../hypre_driver/"))

solver = 3 # Bommer AMG
# max_setup_time = 1000.
# max_solve_time = 1000.
# coeffs_c = "-c 1 1 1 " # specify c-coefficients in format "-c 1 1 1 " 
# coeffs_a = "-a 0 0 0 " # specify a-coefficients in format "-a 1 1 1 " leave as empty string for laplacian and Poisson problems
# problem_name = "-laplacian " # "-difconv " for convection-diffusion problems to include the a coefficients
problem_name = "-difconv "
bmin = 1
bmax = 8
eta = 2
smax = int(np.floor(np.log10(bmax/bmin)/np.log10(eta)))

# define objective function
def objectives(point):
    # task params 
    c_val = point['c_val']
    a_val = point['a_val']
    coeffs_a = str(f'-a {a_val} {a_val} {a_val} ')
    coeffs_c = str(f'-c {c_val} {c_val} {c_val} ')
    
    def budget_map(b, nmin=10, nmax=100):
        k1 = (nmax-nmin)/(bmax-bmin)
        b1 = nmin - k1
        assert k1 * bmax + b1 == nmax
        return int(k1 * b + b1) 
        # return int(45*(np.log(b)/np.log(eta)) + 10)
        # return int(10*b)

    def budget_map3(b, nmin=10, nmax=100):
        k1 = (nmax**3-nmin**3)/(bmax-bmin)
        b1 = nmin**3 - k1
        assert k1 * bmax + b1 == nmax**3
        if b == bmin:
            return nmin
        elif b == bmax:
            return nmax
        else:
            return int((k1 * b + b1)**(1/3)) 
        
    try:
        budget = point['budget']
        # problemsize = int(budget)
        problemsize = budget_map3(budget)
        nx = problemsize
        ny = problemsize
        nz = problemsize
    except:
        nx = budget_map3(bmax)
        ny = budget_map3(bmax)
        nz = budget_map3(bmax)    
    
    # tuning params / input params
    Px = point['Px']
    Py = point['Py']
    Nproc = point['Nproc']
    Pz = int(Nproc / (Px*Py))
    strong_threshold = point['strong_threshold']
    trunc_factor = point['trunc_factor']
    P_max_elmts = point['P_max_elmts']
    coarsen_type = point['coarsen_type']
    relax_type = point['relax_type']
    smooth_type = point['smooth_type']
    smooth_num_levels = point['smooth_num_levels']
    interp_type = point['interp_type']
    agg_num_levels = point['agg_num_levels']
    
    # CoarsTypes = {0:"-cljp ", 1:"-ruge ", 2:"-ruge2b ", 3:"-ruge2b ", 4:"-ruge3c ", 6:"-falgout ", 8:"-pmis ", 10:"-hmis "}
    # CoarsType = CoarsTypes[coarsen_type]
    npernode =  math.ceil(float(Nproc)/nodes)  
    nthreads = int(cores / npernode)

    # call Hypre 
    params = [(nx, ny, nz, coeffs_a, coeffs_c, problem_name, solver,
               Px, Py, Pz, strong_threshold, 
               trunc_factor, P_max_elmts, coarsen_type, relax_type, 
               smooth_type, smooth_num_levels, interp_type, agg_num_levels, nthreads, npernode)]
    
    # if(P_max_elmts==10 and coarsen_type=='6' 
    #     and relax_type=='18' and smooth_type=='6' 
    #     and smooth_num_levels==3 and interp_type=='8' 
    #     and agg_num_levels==1):
    #     return [float("Inf")]
    
    runtime = hypredriver(params, niter=1, JOBID=JOBID)
    print(params, ' hypre time: ', runtime)

    return runtime
    
def models(): # todo
    pass

def main(): 
    global nodes
    global cores
    global JOBID
    global nprocmax
    global nprocmin

    # Parse command line arguments
    args = parse_args()

    amin = args.amin
    amax = args.amax
    cmin = args.cmin
    cmax = args.cmax
    nodes = args.nodes
    cores = args.cores
    nprocmin_pernode = args.nprocmin_pernode
    machine = args.machine
    ntask = args.ntask
    Nloop = args.Nloop
    JOBID = args.jobid
    TUNER_NAME = args.optimization
    TLA = False

    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    # os.system("mkdir -p scalapack-driver/bin/%s; cp ../build/pdqrdriver scalapack-driver/bin/%s/.;" %(machine, machine))

    nprocmax = nodes*cores-1  # YL: there is one proc doing spawning, so nodes*cores should be at least 2
    nprocmin = min(nodes*nprocmin_pernode,nprocmax-1)  # YL: ensure strictly nprocmin<nprocmax, required by the Integer space 

    a_val = Real(amin, amax, transform="normalize", name="a_val")
    c_val = Real(cmin, cmax, transform="normalize", name="c_val")
    Px = Integer(1, nprocmax, transform="normalize", name="Px")
    Py = Integer(1, nprocmax, transform="normalize", name="Py")
    Nproc = Integer(nprocmin, nprocmax, transform="normalize", name="Nproc")
    strong_threshold = Real(0, 1, transform="normalize", name="strong_threshold")
    trunc_factor =  Real(0, 0.999, transform="normalize", name="trunc_factor")
    P_max_elmts = Integer(1, 12,  transform="normalize", name="P_max_elmts")
    # coarsen_type = Categoricalnorm (['0', '1', '2', '3', '4', '6', '8', '10'], transform="onehot", name="coarsen_type")
    coarsen_type = Categoricalnorm (['0', '1', '2', '3', '4', '8', '10'], transform="onehot", name="coarsen_type")
    relax_type = Categoricalnorm (['-1', '0', '6', '8', '16', '18'], transform="onehot", name="relax_type")
    smooth_type = Categoricalnorm (['5', '6', '8', '9'], transform="onehot", name="smooth_type")
    smooth_num_levels = Integer(0, 5,  transform="normalize", name="smooth_num_levels")
    interp_type = Categoricalnorm (['0', '3', '4', '5', '6', '8', '12'], transform="onehot", name="interp_type")
    agg_num_levels = Integer(0, 5,  transform="normalize", name="agg_num_levels")
    r = Real(float("-Inf"), float("Inf"), name="r")
    
    IS = Space([a_val, c_val])
    PS = Space([Px, Py, Nproc, strong_threshold, trunc_factor, P_max_elmts, coarsen_type, relax_type, smooth_type, smooth_num_levels, interp_type, agg_num_levels])
    OS = Space([r])
    
    cst1 = f"Px * Py  <= Nproc"
    cst2 = f"not(P_max_elmts==10 and coarsen_type=='6' and relax_type=='18' and smooth_type=='6' and smooth_num_levels==3 and interp_type=='8' and agg_num_levels==1)"
    constraints = {"cst1": cst1,"cst2": cst2}

    print(IS, PS, OS, constraints)

    problem = TuningProblem(IS, PS, OS, objectives, constraints, None) # no performance model
    computer = Computer(nodes=nodes, cores=cores, hosts=None)

    options = Options()
    options['model_processes'] = 1
    # options['model_threads'] = 1
    options['model_restarts'] = args.Nrestarts
    options['distributed_memory_parallelism'] = False
    options['shared_memory_parallelism'] = False
    # options['mpi_comm'] = None
    options['model_class '] = 'Model_LCM'
    options['verbose'] = False
    options.validate(computer=computer)
    
    options['budget_min'] = bmin
    options['budget_max'] = bmax
    options['budget_base'] = eta
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
    print(f"bmin = {bmin}, bmax = {bmax}, eta = {eta}, smax = {smax}")
    print("samples in one multi-armed bandit loop, NSs_all = ", NSs_all)
    print("total number of samples: ", Ntotal)
    print("total number of evaluations at highest budget: ", Btotal)
    print()
    
    data = Data(problem)
    # giventask = [[(amax-amin)*random.random()+amin,(cmax-cmin)*random.random()+cmin] for i in range(ntask)]
    # giventask = [[0.023181354151175504, 0.6816796355829654]]
    # a, c in [0, 2], 20 tasks
    # giventask = [[1.764404747086545, 0.1690780613092806], [0.04112427493772097, 0.8085715496434904], 
    #              [0.22710987838880214, 1.88151014852883], [1.4295598579456825, 0.6777959247566412], 
    #              [1.2113920952170643, 1.2736211372774706], [1.9506691433602787, 1.6164125044517066], 
    #              [1.0599323074104676, 0.31237589097846064], [0.5539778042983943, 1.7201524881603167], 
    #              [1.1202117146999948, 1.4008498376477942], [1.9764454822916864, 0.49519243217069175],
    #              [1.6242107544285, 1.546469305586815], [0.48556463965870966, 0.9104470584141897], 
    #              [0.24335919069425582, 1.9474734644218745], [1.4548622289600037, 0.44034181519644044], 
    #              [0.30572684587777066, 1.744356822716747], [1.3121087439465606, 1.8289812234496303], 
    #              [0.4368157748180814, 1.7463601854749693], [1.179881068844456, 0.31389817437239453], 
    #              [0.6840593333787133, 1.2532549528173518], [0.808378518110451, 0.7833341762792181]]
    # a, c in [0, 1], 10 tasks
    giventask = [[0.023181354151175504, 0.6816796355829654], [0.15948672790394447, 0.545007082153842], 
                 [0.3061777407959666, 0.2613262568641622], [0.13635922601717265, 0.42633245310802903], 
                 [0.007106878341192613, 0.1275649768730277], [0.14739645464726914, 0.7387029572331887], 
                 [0.03343150589405264, 0.5281563810546017], [0.09379810626184892, 0.4018339729841335],
                 [0.698222888451084, 0.30201670579496664], [0.6026109961485946, 0.07835020207209975]]
    
    NI=len(giventask)
    assert NI == ntask # make sure number of tasks match

    # # the following will use only task lists stored in the pickle file
    # data = Data(problem)


    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        NS = Btotal
        NS1 = max(NS//2, 1)
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=NS1)
        print("stats: ", stats)
        
        # """ Dump the data to file as a new check point """
        # pickle.dump(data, open('Data_nodes_%d_cores_%d_nxmax_%d_nymax_%d_nzmax_%d_machine_%s_jobid_%d.pkl' % (nodes, cores, nxmax, nymax, nzmax, machine, JOBID), 'wb'))
        
        # """ Dump the tuner to file for TLA use """
        # pickle.dump(gt, open('MLA_nodes_%d_cores_%d_nxmax_%d_nymax_%d_nzmax_%d_machine_%s_jobid_%d.pkl' % (nodes, cores, nxmax, nymax, nzmax, machine, JOBID), 'wb'))


        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [a_val, c_val] = [{data.I[tid][0]:.3f}, {data.I[tid][1]:.3f}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

        if TLA is True:
            """ Call TLA for 2 new tasks using the constructed LCM model"""
            newtask = [[0.5, 0.3], [0.2, 1.0]]
            (aprxopts, objval, stats) = gt.TLA1(newtask, NS=None)
            print("stats: ", stats)

            """ Print the optimal parameters and function evaluations"""
            for tid in range(len(newtask)):
                print("new task: %s" % (newtask[tid]))
                print('    predicted Popt: ', aprxopts[tid], ' objval: ', objval[tid])

    
    if(TUNER_NAME=='opentuner'):
        NS = Btotal
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [a_val, c_val] = [{data.I[tid][0]:.3f}, {data.I[tid][1]:.3f}]")
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
            print(f"   [a_val, c_val] = [{data.I[tid][0]:.3f}, {data.I[tid][1]:.3f}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
    
    if(TUNER_NAME=='GPTune_MB'):
        data = Data(problem)
        gt = GPTune_MB(problem, computer=computer, NS=Nloop, options=options)
        (data, stats)=gt.MB_LCM(NS = Nloop, Igiven = giventask)
        print("Tuner: ", TUNER_NAME)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [a_val, c_val] = [{data.I[tid][0]:.3f}, {data.I[tid][1]:.3f}]")
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))
                               
    if(TUNER_NAME=='hpbandster_bandit'):
        NS = Ntotal
        (data,stats)=callhpbandster_bandit.HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, options=options, run_id="hpbandster_bandit", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print(f"   [a_val, c_val] = [{data.I[tid][0]:.3f}, {data.I[tid][1]:.3f}]")
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


def parse_args():
    parser = argparse.ArgumentParser()
    # Problem related arguments
    parser.add_argument('-amin', type=float, default=0, help='min value of coeffs_a')
    parser.add_argument('-amax', type=float, default=2, help='max value of coeffs_a')
    parser.add_argument('-cmin', type=float, default=0, help='min value of coeffs_c')
    parser.add_argument('-cmax', type=float, default=2, help='max value of coeffs_c')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    parser.add_argument('-nprocmin_pernode', type=int, default=1,help='Minimum number of MPIs per machine node for the application code')
    parser.add_argument('-machine', type=str, default='cori1', help='Name of the computer (not hostname)')
    # Algorithm related arguments
    # parser.add_argument('-optimization', type=str, help='Optimization algorithm (opentuner, spearmint, mogpo)')
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-Nloop', type=int, default=-1, help='Number of loops')
    parser.add_argument('-Nrestarts', type=int, default=1, help='Number of model restarts')
    # parser.add_argument('-truns', type=int, default=-1, help='Time of runs')
    # Experiment related arguments
    # 0 means interactive execution (not batch)
    parser.add_argument('-jobid', type=int, default=0, help='ID of the batch job')
    parser.add_argument('-stepid', type=int, default=-1, help='step ID')
    parser.add_argument('-phase', type=int, default=0, help='phase')
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
