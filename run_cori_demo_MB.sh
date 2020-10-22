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

ntask=10
# task='na' # the t value for the task
Nloop=1
# expid=0
# sample_class='SampleLHSMDU' # 'Supported sample classes: SampleLHSMDU, SampleOpenTURNS(default)'

cd examples
rm -rf *.pkl

for expid in 30
do
tuner='opentuner'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

tuner='hpbandster'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}
done

for expid in 31
do
tuner='hpbandster_bandit'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

tuner='GPTune_MB'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner}  2>&1 | tee a.out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}
done


# TEST multi-loop
# testid=4
# tuner='GPTune_MB'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -ntask ${ntask} -nbracket ${nbracket} -optimization ${tuner}  2>&1 | tee a.out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_testid${testid}


# TEST all random config sampling in GPTune
# testid=4
# ntask=1
# tuner='GPTune_MB'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner}  2>&1 | tee a.out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_testid${testid}
