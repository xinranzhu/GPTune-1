#!/bin/bash
#SBATCH -C knl,quad,cache
#SBATCH -J exp_hypre
#SBATCH --qos=regular
#SBATCH -t 08:00:00
#SBATCH --nodes=8
#SBATCH --mail-user=xz584@cornell.edu
#SBATCH --mail-type=ALL

module unload darshan
module swap craype-haswell craype-mic-knl
module load craype-hugepages2M
module unload cray-libsci

module load python/3.7-anaconda-2019.10
module unload cray-mpich

module swap PrgEnv-intel PrgEnv-gnu
# export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/global/homes/x/xinranz/projects/GPTune/scalapack-2.1.0/build/lib
module load openmpi/4.0.1
export OMPI_MCA_btl_ugni_virtual_device_count=1
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

nxmin=100
nymin=100
nzmin=100
nxmax=200
nymax=200
nzmax=200
ntask=30
nrun=10
nodes=1
cores=64
nprocmin_pernode=64  # nprocmin_pernode=cores means flat MPI 



# test hypredriver, the following calling sequence will first dump the data to file when using GPTune, then read data when using opentuner or hpbandster to make sure they use the same tasks as GPTune
cd examples
rm -rf *.pkl
tuner='GPTune'
mpirun --oversubscribe -N 64 --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee | tee a.out_hypre_ML_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}
tuner='opentuner'
mpirun --oversubscribe -N 64 --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee | tee a.out_hypre_ML_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}
tuner='hpbandster'
mpirun --oversubscribe -N 64 --mca pmix_server_max_wait 3600 --mca pmix_base_exchange_timeout 3600 --mca orte_abort_timeout 3600 --mca plm_rsh_no_tree_spawn true -n 1 python ./hypre.py -nxmax ${nxmax} -nymax ${nymax} -nzmax ${nzmax} -nxmin ${nxmin} -nymin ${nymin} -nzmin ${nzmin} -nodes ${nodes} -cores ${cores} -ntask ${ntask} -nrun ${nrun} -machine cori1 -jobid 0 -optimization ${tuner}   2>&1 | tee | tee a.out_hypre_ML_nxmax${nxmax}_nymax${nymax}_nzmax${nzmax}_nxmin${nxmin}_nymin${nymin}_nzmin${nzmin}_nodes${nodes}_core${cores}_ntask${ntask}_nrun${nrun}_${tuner}
