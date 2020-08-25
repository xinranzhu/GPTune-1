import sys, os, re
import numpy as np
import time
import argparse
import pickle
from random import *
import math
from hypredriver import hypredriver

sys.path.insert(0, os.path.abspath(__file__ + "/../hypre_driver/"))

solver = 3 # Bommer AMG
# max_setup_time = 1000.
# max_solve_time = 1000.
problem_name = "-difconv " # "-difconv " for convection-diffusion problems to include the a coefficients

def main():
    args = parse_args()
    # JOBID = args.jobid
    TUNER_NAME = args.optimization
    machine = args.machine
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    # params = [(nx, ny, nz, coeffs_a, coeffs_c, problem_name, solver,
                #    Px, Py, Pz, strong_threshold, 
                #    trunc_factor, P_max_elmts, coarsen_type, relax_type, 
                #    smooth_type, smooth_num_levels, interp_type, agg_num_levels, nthreads, npernode)]
                       
    c_val = 0.8
    a_val = 0.6
    coeffs_a = str(f'-a {a_val} {a_val} {a_val} ')
    coeffs_c = str(f'-c {c_val} {c_val} {c_val} ')
    print("PDE coeffs: ", coeffs_a, coeffs_c)
    problem_size = np.arange(10,150,10)
    params = [('-difconv ', 3, 19, 1, 1, 0.3212934323637363, 0.9471727354342201, 2, '0', '18', '9', 4, '0', 3, 1, 30),
              ('-difconv ', 3, 8, 1, 3, 0.46207559273813614, 0.16093209390282756, 12, '10', '6', '9', 5, '12', 3, 1, 31),
              ('-difconv ', 3, 13, 1, 2, 0.8776120880938103, 0.09890238431809178, 2, '0', '0', '8', 0, '12', 2, 1, 30)]
    runtime_set = []
    for i in range(len(params)):  
        runtime_set_i = [] 
        print("param id: ", i)
        for nx in problem_size:
            params_temp = [(nx, nx, nx) + (coeffs_a, coeffs_c) + params[i]]
            runtime = hypredriver(params_temp, niter=3, JOBID=0)
            runtime_set_i.append(runtime)
            print(params_temp, ' hypre time: ', runtime)
        runtime_set.append(runtime_set_i)
    
    print("runtime_set: ", runtime_set)
    
    # test budget
    # params_task0 = [[(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 19, 1, 1, 0.3212934323637363, 0.9471727354342201, 2, '0', '18', '9', 4, '0', 3, 1, 30)], 
    #                 [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 8, 1, 3, 0.46207559273813614, 0.16093209390282756, 12, '10', '6', '9', 5, '12', 3, 1, 31)],
    #                 [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 13, 1, 2, 0.8776120880938103, 0.09890238431809178, 2, '0', '0', '8', 0, '12', 2, 1, 30)],
    #                 [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 6, 2, 0.25964243246187946, 0.4691071798485803, 12, '10', '0', '6', 5, '0', 1, 1, 30)],
    #                 [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 15, 1, 2, 0.4102633159440464, 0.9040766195471719, 3, '1', '18', '9', 3, '12', 5, 1, 31)],
    #                 [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 5, 3, 0.8837338741015095, 0.22131277713813266, 2, '8', '6', '6', 4, '6', 3, 1, 31)],
    #          ]
    # params_task1 = [[(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 15, 2, 0.7067388180937928, 0.8085763444748442, 9, '10', '0', '5', 0, '0', 0, 1, 31)],
    #                 [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 1, 10, 6, 0.7588895669433341, 0.8934521477475993, 3, '10', '16', '6', 3, '8', 5, 1, 32)], 
    #                 [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 12, 2, 2, 0.5781978010046213, 0.8936710067146584, 10, '0', '-1', '8', 1, '12', 0, 1, 32)],
    #                 [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 32, 1, 1, 0.4213398520374657, 0.017226016401326844, 2, '8', '8', '6', 5, '3', 4, 1, 32)],
    #                 [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 1, 39, 1, 0.32807267072175905, 0.475729880258016, 10, '2', '0', '8', 5, '6', 5, 1, 31)],
    #                 [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 4, 7, 2, 0.12148916017850653, 0.6754905278349941, 7, '10', '8', '9', 0, '6', 1, 1, 32)]]
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')
    # parser.add_argument('-max_iter', type=str, default='1000', help='maximum number of iteration in hypre solver, test for budget setup')
    # parser.add_argument('-tol', type=str, default='1e-8', help='tolerance in hypre driver, test for budget setup')
    # parser.add_argument('-paramid', type=int, default=0, help='index of the testing params')
    # parser.add_argument('-taskid', type=int, default=0, help='index of testing task')
    # parser.add_argument('-budget', type=float, default=10, help='budget for the hypre task')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
