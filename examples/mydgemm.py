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

"""
import sys, os, re

# add GPTunde path in front of all python pkg path
sys.path.insert(0, os.path.abspath(__file__ + "/../../GPTune/"))
sys.path.insert(0, os.path.abspath(__file__ + "/../dgemm/dgemm-driver"))

# from dgemmdriver import dgemmdriver
import dgemmdriver
from autotune.search import *
from autotune.space import *
from autotune.problem import *
from gptune import GPTune
from data import Data
from data import Categoricalnorm
from options import Options
from computer import Computer
import numpy as np
import time
import argparse
import pickle
from random import *
from callopentuner import OpenTuner
from callhpbandster import HpBandSter
import math

# import mpi4py
# from mpi4py import MPI

from dgemmdriver import dgemmdriver

# define objective function
def objectives(point):
    # task params 
    matrix_size = point['matrix_size']
    
    # tuning params / input params
    fastmath = point['fastmath']
    marchnative = point['marchnative']
    ftreevectorize = point['ftreevectorize']
    funrollloops = point['funrollloops']
    
    # call Hypre 
    # can also include task param here, like matrix size!
    params = [(matrix_size, fastmath, marchnative, ftreevectorize, funrollloops)]
  
    mflop = dgemmdriver(params)

    return mflop
    
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

    matrix_size_min = args.matrix_size_min
    matrix_size_max = args.matrix_size_max
    nodes = 1
    cores = 2

    ntask = args.ntask
    nruns = args.nruns
    TUNER_NAME = args.optimization
    TLA = False

    nprocmax = 1 # YL: there is one proc doing spawning, so nodes*cores should be at least 2
    nprocmin = 1  # YL: ensure strictly nprocmin<nprocmax, required by the Integer space 

    
    matrix_size = Integer(matrix_size_min, matrix_size_max, transform="normalize", name="matrix_size")
    fastmath = Categoricalnorm(["0", "fastmath"], transform="onehot", name="fastmath")
    marchnative = Categoricalnorm(["0", "marchnative"], transform="onehot", name="marchnative")
    ftreevectorize = Categoricalnorm(["0", "ftreevectorize"], transform="onehot", name="ftreevectorize")
    funrollloops = Categoricalnorm(["0", "funrollloops"], transform="onehot", name="funrollloops")
    mflop = Real(float(0), float("Inf"), name="mflop")
    
    IS = Space([matrix_size])
    PS = Space([fastmath, marchnative, ftreevectorize, funrollloops])
    OS = Space([mflop])
    
    constraints = {}

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
    # options['model_class '] = 'Model_GPy_LCM'
    options['verbose'] = False
    options.validate(computer=computer)
    
    
    """ Intialize the tuner with existing data stored as last check point"""
   

    giventask = [[31], [32], [96], [97], [127], [128], [129], [191], [192], [229]]
    # giventask = [[31], [32], [96]]
    assert ntask == len(giventask)
    # # the following will use only task lists stored in the pickle file
    data = Data(problem)


    if(TUNER_NAME=='GPTune'):
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        NI = len(giventask)
        NS = nruns
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    matrix_size %d" % data.I[tid][0])
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    
    if(TUNER_NAME=='opentuner'):
        NI = ntask
        NS = nruns
        (data,stats) = OpenTuner(T=giventask, NS=NS, tp=problem, computer=computer, run_id="OpenTuner", niter=1, technique=None)
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    matrix_size %d" % data.I[tid][0])
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if(TUNER_NAME=='hpbandster'):
        NI = ntask
        NS = nruns
        (data,stats)=HpBandSter(T=giventask, NS=NS, tp=problem, computer=computer, run_id="HpBandSter", niter=1)
        print("stats: ", stats)
        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    matrix_size %d" % data.I[tid][0])
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid].tolist())
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

    if TUNER_NAME.startswith('GPTune_multistart'):
        data = Data(problem)
        options['model_restarts'] = int(TUNER_NAME[17:])
        print(f"Multistart in GPTune = {options['model_restarts']}")
        gt = GPTune(problem, computer=computer, data=data, options=options, driverabspath=os.path.abspath(__file__))        
        """ Building MLA with the given list of tasks """
        NI = len(giventask)
        NS = nruns
        (data, model, stats) = gt.MLA(NS=NS, NI=NI, Igiven=giventask, NS1=max(NS//2, 1))
        print("stats: ", stats)

        """ Print all input and parameter samples """
        for tid in range(NI):
            print("tid: %d" % (tid))
            print("    matrix_size %d" % data.I[tid][0])
            print("    Ps ", data.P[tid])
            print("    Os ", data.O[tid])
            print('    Popt ', data.P[tid][np.argmin(data.O[tid])], 'Oopt ', min(data.O[tid])[0], 'nth ', np.argmin(data.O[tid]))

def parse_args():
    parser = argparse.ArgumentParser()
    # Problem related arguments
    parser.add_argument('-matrix_size_min', type=int, default=1, help='minimum matrix size')
    parser.add_argument('-matrix_size_max', type=int, default=1600, help='maximum matrix size')
    # Machine related arguments
    parser.add_argument('-nodes', type=int, default=1, help='Number of machine nodes')
    parser.add_argument('-cores', type=int, default=1, help='Number of cores per machine node')
    # Algorithm related arguments
    parser.add_argument('-optimization', type=str,default='GPTune',help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-ntask', type=int, default=-1, help='Number of tasks')
    parser.add_argument('-nruns', type=int, default=-1, help='Number of runs per task')
    parser.add_argument('-Nrestarts', type=int, default=1, help='Number of model restarts')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
