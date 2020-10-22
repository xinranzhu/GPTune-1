import numpy as np
import os, sys, re
import pandas as pd


# Paths
ROOTDIR = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
# EXPDIR = os.path.abspath(os.path.join(ROOTDIR, "dgemm-driver/exp", MACHINE_NAME + '/' + TUNER_NAME))
EXCUDIR = os.path.abspath(os.path.join(ROOTDIR, "dgemm-driver/src/cs5220-project1/run_mydgemm.sh"))
OUTDIR =  os.path.abspath(os.path.join(ROOTDIR, "dgemm-driver/src/cs5220-project1/timing-mine.csv"))

def execute(param):
    # extract arguments
    matrix_size = param['matrix_size']
    # matrix_size_set = []
    # if matrix_size == 1:
    #     matrix_size_set = [31, 32, 96, 97, 127, 128, 129, 191, 192, 229]
    # elif matrix_size == 2:
    #     matrix_size_set = [31, 40]
    #     # matrix_size_set = [31, 32, 96, 97, 127, 128, 129, 191, 192, 229]
    #     # matrix_size_set = [255, 256, 257, 319, 320, 321, 417, 479, 480, 511, 512, 639, 640,
    #     #                     767, 768, 769, 1023, 1024, 1025, 1525, 1526, 1527]


    # Opt_level = param['Opt_level'] 
    fastmath = param['fastmath'] 
    marchnative = param['marchnative']
    ftreevectorize = param['ftreevectorize']
    funrollloops = param['funrollloops']
    fallowstoredataraces = param['fallowstoredataraces']

    mflop_set = []
    
    def read_output(outputfilename):
        df = pd.read_csv(outputfilename)
        tasksize = df['size'][0]
        mflop = df['mflop'][0]/1e3
        print(f"[----- tasksize = {tasksize}, mflop = {mflop} -----]\n")
        return mflop 

    # for i in range(len(matrix_size_set)):
    os_command = EXCUDIR + f" {matrix_size}"

    flags = " "
    # if Opt_level == 1: 
    #     flags = flags + " -O3"
    # elif Opt_level == 2:
    #     flags = flags + " -O3"
    # elif Opt_level == 3:
    #     flags = flags + " -O3"
    # elif Opt_level == 4:
    #     flags = flags + " -Os"

    if fastmath == b"fastmath":
        # os_command = "dgemm-driver/src/cs5220-project1/run_mydgemm.sh -fastmath"
        flags = flags + " -ffast-math"
    
    # if other flag exists... -march=native -ftree-vectorize -funroll-loops -ffast-math
    if marchnative == b"marchnative":
        flags = flags + " -march=native"
    
    if ftreevectorize == b"ftreevectorize":
        flags = flags + " -ftree-vectorize"

    if funrollloops == b"funrollloops":
        flags = flags + " -funroll-loops"
    
    if fallowstoredataraces == b"fallowstoredataraces":
        flags = flags + " -fallow-store-data-races"

    os_command = os_command + flags
    print(f"os_command = {os_command} \n")
    os.system(os_command)

    mflop = read_output(OUTDIR)
    # mflop_set.append(-mflop)

    # print(f"mflop_set = {mflop_set} \n\n")
    return -mflop



def dgemmdriver(params):

    global ROOTDIR
    global EXCUDIR
    # can also include task param here, like matrix size!
    dtype = [('matrix_size', '<i8'), ('fastmath', '|S8'), 
            ('marchnative', '|S11'), ('ftreevectorize', '|S14'), ('funrollloops', '|S12'),
            ('fallowstoredataraces',  '|S20')]
    params = np.array(params, dtype=dtype)
    mflop_set = []
    for param in params:
        print(f"Current param {param}\n")
        mflop_cur = execute(param)
        mflop_set.append(mflop_cur)
    # os.system('rm -fr %s'%(RUNDIR))
    return mflop_set


if __name__ == "__main__":


    # os.environ['MACHINE_NAME'] = 'cori'
    # os.environ['TUNER_NAME'] = 'GPTune'
    # params = [(60, 50, 80, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 2, 2, 0.25, 0, 4, 10, 8, 6, 0, 6, 0, 1, 1),\
    #           (60, 50, 80, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 2, 2, 0.3, 0.2, 5, 10, 8, 6, 1, 6, 1, 1, 1)
    #           ]
    # times = hypredriver(params, niter=1)

    params = [(30, 1, 1, 1, 1), (30, 1, 0, 0, 0), (30, 0, 0, 0, 0)]
    # params = [(30, 1,0,0,0)]
    mflop_set = dgemmdriver(params)

    print('mflop_set = ', mflop_set)
