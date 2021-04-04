#!/bin/bash
##SBATCH -J test_driver  
#SBATCH --partition=bindel-cpu
##SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -n 2
##SBATCH --tasks-per-node=8
##SBATCH --cpus-per-task=4

#SBATCH -o /home/xz584/projects/GPTune/outputs/%j_o.txt
#SBATCH -e /home/xz584/projects/GPTune/outputs/%j_e.txt

#OpenMP settings:
# export OMP_NUM_THREADS=4
export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

. /home/xz584/anaconda3/etc/profile.d/conda.sh
conda activate gptune2

export MKLROOT=/home/xz584/anaconda3/envs/gptune2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKLROOT}/lib

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
# module load openmpi/4.0.1

export PYTHONPATH=/home/xz584/anaconda3/envs/gptune2/lib/python3.7/site-packages
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

cd examples

lscpu


nodes=1
cores=2
ntask=1
dim_task=1
Nrestarts=5
nruns=20

expid='VerD-2'
tuner='GPTune'
mpirun -n 1 python -u mydgemm.py -dim_task ${dim_task} -nodes ${nodes} -Nrestarts ${Nrestarts} -cores ${cores} -ntask ${ntask} -nruns ${nruns} -optimization ${tuner} 2>&1 | tee a.ntask${ntask}_expid${expid}


# for expid in {0..4}
# do
# tuner='GPTune'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.graphite_out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='GPTune_MB'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner}  2>&1 | tee a.graphite_out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='hpbandster_bandit'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.graphite_out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='opentuner'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.graphite_out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='hpbandster'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python -u demo_MB.py -nodes ${nodes} -cores ${cores} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.graphite_out_demo_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# done




