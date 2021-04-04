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
coeffs_c = "-c 1 1 1 " # specify c-coefficients in format "-c 1 1 1 " 
coeffs_a = "-a 0 0 0 " # specify a-coefficients in format "-a 1 1 1 " leave as empty string for laplacian and Poisson problems
problem_name = "-laplacian " # "-difconv " for convection-diffusion problems to include the a coefficients

def main():
    args = parse_args()
    JOBID = args.jobid
    TUNER_NAME = args.optimization
    machine = args.machine
    os.environ['MACHINE_NAME'] = machine
    os.environ['TUNER_NAME'] = TUNER_NAME
    # params = [(nx, ny, nz, coeffs_a, coeffs_c, problem_name, solver,
                #    Px, Py, Pz, strong_threshold, 
                #    trunc_factor, P_max_elmts, coarsen_type, relax_type, 
                #    smooth_type, smooth_num_levels, interp_type, agg_num_levels, nthreads, npernode)]

    # failed3 params, Nproc = 475
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 19, 5, 5, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # original problematic one
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 15, 5, 5, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 12, 5, 5, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 11, 5, 5, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 10, 5, 5, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # works, 250
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 5, 5, 5, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # works
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 10, 6, 5, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 20, 20, 1, 0.6243, 0.699, 9, 0, 6, 7, 2, 6, 3, 1, 32)] 
    # failed
    
    
    # failed4 params
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 1, 298, 1, 0.44097267, 0.70830443, 2, 0, 8, 7, 3, 5, 5, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 149, 1, 0.44097267, 0.70830443, 2, 0, 8, 7, 3, 5, 5, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 50, 2, 0.44097267, 0.70830443, 2, 0, 8, 7, 3, 5, 5, 1, 32)] 
    # worked
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 16, 9, 0.44097267, 0.70830443, 2, 0, 8, 7, 3, 5, 5, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 60, 2, 0.44097267, 0.70830443, 2, 0, 8, 7, 3, 5, 5, 1, 32)] 
    # worked
    
    
    # failed5 params
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 177, 1, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # failed
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 128, 1, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # worked, 256
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 129, 1, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # worked, 258
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 130, 1, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # failed, 260
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 1, 259, 1, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # failed 259
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 43, 3, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # worked, 258
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 43, 3, 2, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # worked, 258
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 65, 2, 0.30017979, 0.80794053, 12, 3, 6, 7, 5, 6, 2, 1, 32)] 
    # failed, 260
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 1, 289, 1, 0.17927643772415036, 0.4794427825943006, 7, '10', '0', '6', 1, '6', 2, 1, 32)]  
    # worked, 289
    
    # params = [(194, 168, 157, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 162, 4, 1, 0.15118518819041837, 0.16462433400943832, 10, 6, 18, 6, 3, 8, 1, 1, 32)] 
    # failed
    # params = [(194, 168, 157, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 81, 4, 2, 0.15118518819041837, 0.16462433400943832, 10, 6, 18, 6, 3, 8, 1, 1, 32)]
    # failed 
    # params = [(194, 168, 157, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 8, 4, 2, 0.15118518819041837, 0.16462433400943832, 10, 6, 18, 6, 3, 8, 1, 1, 32)] 
    # failed

    # params = [(172, 175, 159, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 62, 2, 2, 0.9941, 0.348, 6, 4, 18, 6, 2, 3, 1, 1, 32)] 
    # worked
    # params = [(173, 188, 188, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 62, 1, 4, 0.7184, 0.5767, 4, 1, 6, 8, 1, 3, 5, 1, 32)]
    # worked 
    # params = [(194, 168, 157, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 8, 30, 1, 0.4115, 0.22, 12, 1, 16, 6, 4, 6, 1, 1, 32)]
    # worked
    # params = [(184, 165, 153, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 1, 94, 2, 0., 0., 12, 1, 6, 5, 0, 12, 0, 1, 32)]
    # worked, opentuner
    # params = [(196, 166, 189, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian', 3, 2, 33, 3, 0.54664604, 0.3110243, 9, 3, -1, 6, 4, 6, 3, 1, 32)]
    # worked, hpbandster
    
    # test budget
    params_task0 = [[(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 19, 1, 1, 0.3212934323637363, 0.9471727354342201, 2, '0', '18', '9', 4, '0', 3, 1, 30)], 
                    [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 8, 1, 3, 0.46207559273813614, 0.16093209390282756, 12, '10', '6', '9', 5, '12', 3, 1, 31)],
                    [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 13, 1, 2, 0.8776120880938103, 0.09890238431809178, 2, '0', '0', '8', 0, '12', 2, 1, 30)],
                    [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 6, 2, 0.25964243246187946, 0.4691071798485803, 12, '10', '0', '6', 5, '0', 1, 1, 30)],
                    [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 15, 1, 2, 0.4102633159440464, 0.9040766195471719, 3, '1', '18', '9', 3, '12', 5, 1, 31)],
                    [(55, 66, 70, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 5, 3, 0.8837338741015095, 0.22131277713813266, 2, '8', '6', '6', 4, '6', 3, 1, 31)],
             ]
    params_task1 = [[(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 15, 2, 0.7067388180937928, 0.8085763444748442, 9, '10', '0', '5', 0, '0', 0, 1, 31)],
                    [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 1, 10, 6, 0.7588895669433341, 0.8934521477475993, 3, '10', '16', '6', 3, '8', 5, 1, 32)], 
                    [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 12, 2, 2, 0.5781978010046213, 0.8936710067146584, 10, '0', '-1', '8', 1, '12', 0, 1, 32)],
                    [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 32, 1, 1, 0.4213398520374657, 0.017226016401326844, 2, '8', '8', '6', 5, '3', 4, 1, 32)],
                    [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 1, 39, 1, 0.32807267072175905, 0.475729880258016, 10, '2', '0', '8', 5, '6', 5, 1, 31)],
                    [(100, 100, 100, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 4, 7, 2, 0.12148916017850653, 0.6754905278349941, 7, '10', '8', '9', 0, '6', 1, 1, 32)]]
    params_task2 = []
    params_task3 = []
    params_all = [params_task0, params_task1, params_task2, params_task3]
    runtime = hypredriver(params_all[args.taskid][args.paramid], niter=3, JOBID=0, max_iter=args.max_iter, tol=args.tol, budget=args.budget)
    print(params_all[args.taskid][args.paramid], ' hypre time: ', runtime)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-machine', type=str, help='Name of the computer (not hostname)')
    parser.add_argument('-optimization', type=str,default='GPTune', help='Optimization algorithm (opentuner, hpbandster, GPTune)')
    parser.add_argument('-jobid', type=int, default=-1, help='ID of the batch job')
    parser.add_argument('-max_iter', type=str, default='1000', help='maximum number of iteration in hypre solver, test for budget setup')
    parser.add_argument('-tol', type=str, default='1e-8', help='tolerance in hypre driver, test for budget setup')
    parser.add_argument('-paramid', type=int, default=0, help='index of the testing params')
    parser.add_argument('-taskid', type=int, default=0, help='index of testing task')
    parser.add_argument('-budget', type=float, default=10, help='budget for the hypre task')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
