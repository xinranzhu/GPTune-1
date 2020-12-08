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

nodes=3
cores=32
nprocmin_pernode=32  # nprocmin_pernode=cores means flat MPI 

# test hypredriver, the following calling sequence will first dump the data to file when using GPTune, then read data when using opentuner or hpbandster to make sure they use the same tasks as GPTune
cd examples
rm -rf *.pkl


amin=0
amax=1
cmin=0
cmax=1


Nloop=1
# expid='TEST-Corr'
# tuner='GPTune'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} 2>&1 | tee a.out_hypre_MB_amin${amin}_cmin${cmin}_amax${amax}_cmax${cmax}_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expid}






        # giventask = [[0.2, 0.5], [0.159, 0.545], 
        #             [0.02, 0.682], [0.9, 0.02], 
        #             [0.7, 0.3], [0.147, 0.739], 
        #             [0.033, 0.528], [0.094, 0.402],
        #             [0.698, 0.302], [0.603, 0.0784]]

ntask=1
nrun=50
expname='Rtrue_R30-R34-LHS'

taskid=0
a=0.2
c=0.8
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}

taskid=1
a=0.159
c=0.545
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}


taskid=2
a=0.02
c=0.682
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}


taskid=3
a=0.9
c=0.02
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}

taskid=4
a=0.7
c=0.3
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}


taskid=5
a=0.147
c=0.739
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}

taskid=6
a=0.033
c=0.528
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}

taskid=7
a=0.094
c=0.402
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}

taskid=8
a=0.698
c=0.302
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}

taskid=9
a=0.603
c=0.0784
tuner='GPTune'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_MB_Rtrue.py -amin ${amin} -amax ${amax} -cmin ${cmin} -cmax ${cmax} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -a ${a} -c ${c} -nrun ${nrun} 2>&1 | tee a.out_hypre_MB_ntask${ntask}_Nloop${Nloop}_${tuner}_expid${expname}-${taskid}
