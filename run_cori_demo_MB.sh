#!/bin/bash
#SBATCH -C haswell
#SBATCH -J test_driver
#SBATCH --qos=regular
#SBATCH -t 02:00:00
#SBATCH --nodes=2
#SBATCH --mail-user=xz584@cornell.edu
#SBATCH --mail-type=ALL
##SBATCH --ntasks=8
##SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=8

#OpenMP settings:
# export OMP_NUM_THREADS=4
export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64:/opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
module load openmpi/4.0.1

export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/scalapack-driver/spt/
export PYTHONPATH=$PYTHONPATH:$PWD/examples/hypre-driver/
export PYTHONWARNINGS=ignore

CCC=mpicc
CCCPP=mpicxx
FTN=mpif90

# task='na' # the t value for the task
Nloop=1
# sample_class='SampleLHSMDU' # 'Supported sample classes: SampleLHSMDU, SampleOpenTURNS(default)'

cd examples
rm -rf *.pkl

nodes=1
cores=32
ntask=2
plot=0
restart=1
perfmodel=0
expid='0'
tuner='GPTune'
mpirun -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart} -perfmodel ${perfmodel} -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

# tuner='GPTuneBand'
# mpirun -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart} -perfmodel ${perfmodel} -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

# tuner='hpbandster'
# mpirun -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart} -perfmodel ${perfmodel} -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

# tuner='TPE'
# mpirun -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart} -perfmodel ${perfmodel} -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}
