#!/bin/bash
##SBATCH -J test_driver  
#SBATCH --partition=bindel-cpu
##SBATCH -t 04:00:00
#SBATCH -N 2
#SBATCH -n 16
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
rm -rf *.pkl

nodes=1
cores=8
nprocmin_pernode=8  # nprocmin_pernode=cores means flat MPI 
Nloop=1

ntask=10
amin=0
amax=1
cmin=0
cmax=1

for expid in {0..2}
do
# tuner='GPTune_MB'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner}  2>&1 | tee a.graphite_out_hypre_MB_amin${amin}_cmin${cmin}_amax${amax}_cmax${cmax}_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='GPTune'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner}  2>&1 | tee a.graphite_out_hypre_MB_amin${amin}_cmin${cmin}_amax${amax}_cmax${cmax}_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='opentuner'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner}  2>&1 | tee a.graphite_out_hypre_MB_amin${amin}_cmin${cmin}_amax${amax}_cmax${cmax}_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='hpbandster_bandit'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.graphite_out_hypre_MB_amin${amin}_cmin${cmin}_amax${amax}_cmax${cmax}_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}

# tuner='hpbandster'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} 2>&1 | tee a.graphite_out_hypre_MB_amin${amin}_cmin${cmin}_amax${amax}_cmax${cmax}_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}
done


expid='TestGraphite'
tuner='GPTune_MB'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner}  2>&1 | tee a.graphite_out_hypre_MB_amin${amin}_cmin${cmin}_amax${amax}_cmax${cmax}_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}
