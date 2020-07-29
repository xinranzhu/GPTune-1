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

nxmin=10
nymin=10
nzmin=10
nxmax=100
nymax=100
nzmax=100
nodes=2
cores=32
nprocmin_pernode=32  # nprocmin_pernode=cores means flat MPI 
nrun=10
ntask=30


# test hypredriver, the following calling sequence will first dump the data to file when using GPTune, then read data when using opentuner or hpbandster to make sure they use the same tasks as GPTune
cd examples
rm -rf *.pkl

# MLA
# equation='covdiff'
# tuner='GPTune'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_${equation}_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}
# tuner='GPTune_multistart5'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_${equation}_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}
# tuner='opentuner'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_${equation}_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}
# tuner='hpbandster'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_${equation}_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}


# Single task tuning
# ntask=1
# tasksize=200
# tuner='GPTune'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_covdiff_nodes${nodes}_core${cores}_tasksize${tasksize}_nrun${nrun}_${tuner}
# tuner='hpbandster'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_covdiff_nodes${nodes}_core${cores}_tasksize${tasksize}_nrun${nrun}_${tuner}

# debug, test budget
# ntask=1
# tuner='test_budget'
# taskid=0
# paramid=0
# tol='1e-7'
# max_iter='1000'
# budget=1
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_debug.py -budget ${budget} -max_iter ${max_iter} -tol ${tol} -paramid ${paramid} -taskid ${taskid} -machine cori1 -jobid 0 -optimization ${tuner} 2>&1 | tee a.test_budget_task${taskid}_param${paramid}_budget${budget}

# test GPTune
# nodes=1
# ntask=1
# tuner='GPTune'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.out_hypre_ML_test_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}

# test bandit in hyperbandster
# ntask=30
# nodes=1
# nrun=20
# tuner='hpbandster_bandit'
# mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre_bandit.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner} 2>&1 | tee a.out_hypre_ML_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}

# test opentuner
nodes=1
ntask=1
nrun=5
tuner='opentuner'
mpirun --oversubscribe --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -nprocmin_pernode ${nprocmin_pernode} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee a.testopentuner_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}
