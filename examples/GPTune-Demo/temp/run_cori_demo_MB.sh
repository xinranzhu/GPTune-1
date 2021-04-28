#!/bin/bash -l
#SBATCH -q premium
#SBATCH -N 9
#SBATCH -t 10:00:00
#SBATCH -J GPTune_nimrod
#SBATCH --mail-user=liuyangzhuan@lbl.gov
#SBATCH -C haswell
cd ../../

ModuleEnv='cori-haswell-openmpi-gnu'

############### Cori Haswell Openmpi+GNU
if [ $ModuleEnv = 'cori-haswell-openmpi-gnu' ]; then
    # module load python/3.7-anaconda-2019.10
    # module unload cray-mpich
    # module swap PrgEnv-intel PrgEnv-gnu
    # module load openmpi/4.0.1
    # export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    # export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages

    module load gcc/8.3.0
    module unload cray-mpich
    module unload openmpi
    module unload PrgEnv-intel
    module load PrgEnv-gnu
    module load openmpi/4.0.1
    module unload craype-hugepages2M
    module unload cray-libsci
    module unload atp    
    module load python/3.7-anaconda-2019.10
    export MKLROOT=/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=~/.local/cori/3.7-anaconda-2019.10/lib/python3.7/site-packages

    module unload python
    USER="$(basename $HOME)"
    PREFIX_PATH=/global/cscratch1/sd/$USER/conda/pytorch/1.8.0
    source /usr/common/software/python/3.7-anaconda-2019.10/etc/profile.d/conda.sh
    conda activate $PREFIX_PATH
    export MKLROOT=$PREFIX_PATH
    BLAS_INC="-I${MKLROOT}/include"
    export LD_LIBRARY_PATH=$PREFIX_PATH/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/examples/SuperLU_DIST/superlu_dist/parmetis-4.0.3/install/lib/
    export PYTHONPATH=$PREFIX_PATH/lib/python3.7/site-packages


    proc=haswell
    cores=32
    machine=cori
    software_json=$(echo ",\"software_configuration\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
    loadable_software_json=$(echo ",\"loadable_software_configurations\":{\"openmpi\":{\"version_split\": [4,0,1]},\"scalapack\":{\"version_split\": [2,1,0]},\"gcc\":{\"version_split\": [8,3,0]}}")
else
    echo "Untested ModuleEnv: $ModuleEnv, please add the corresponding definitions in this file"
    exit
fi    
###############


export GPTUNEROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/mpi4py/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPTune/
export PYTHONWARNINGS=ignore

cd -

nodes=1  # number of nodes to be used
machine_json=$(echo ",\"machine_configuration\":{\"machine_name\":\"$machine\",\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}")
loadable_machine_json=$(echo ",\"loadable_machine_configurations\":{\"$machine\":{\"$proc\":{\"nodes\":$nodes,\"cores\":$cores}}}")
tp=GPTune-Demo
app_json=$(echo "{\"tuning_problem_name\":\"$tp\"")
echo "$app_json$machine_json$software_json$loadable_machine_json$loadable_software_json}" | jq '.' > .gptune/meta.json


# sample_class='SampleLHSMDU' # 'Supported sample classes: SampleLHSMDU, SampleOpenTURNS(default)'


Nloop=2
ntask=1
plot=0
restart=1
expid='TEST'
rm gptune.db/GPTune-Demo.json

tuner='GPTune'
LD_PRELOAD=/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_core.so:/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_sequential.so \
mpirun -n 1 python -u demo_MB.py  -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart}  -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

rm gptune.db/GPTune-Demo.json

tuner='GPTuneBand'
LD_PRELOAD=/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_core.so:/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_sequential.so \
mpirun -n 1 python -u demo_MB.py  -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart}  -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

# tuner='hpbandster'
# LD_PRELOAD=/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_core.so:/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_sequential.so \
# mpirun -n 1 python -u demo_MB.py  -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart}  -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

# tuner='TPE'
# LD_PRELOAD=/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_core.so:/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_sequential.so \
# mpirun -n 1 python -u demo_MB.py  -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart}  -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}

# tuner='opentuner'
# LD_PRELOAD=/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_core.so:/global/cscratch1/sd/xinranz/conda/pytorch/1.8.0/lib/libmkl_sequential.so \
# mpirun -n 1 python -u demo_MB.py  -ntask ${ntask} -Nloop ${Nloop} -optimization ${tuner} -restart ${restart}  -plot ${plot} -expid ${expid} 2>&1 | tee a.out_demo_ntask${ntask}_nruns${nruns}_expid${expid}_${tuner}
