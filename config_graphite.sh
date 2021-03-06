#!/bin/bash
set -x
set -e

. /home/xz584/anaconda3/etc/profile.d/conda.sh
conda activate gptune2

export MKLROOT=/home/xz584/anaconda3/envs/gptune2/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MKLROOT}/lib

# module use /global/common/software/m3169/cori/modulefiles
# module unload openmpi
# module load openmpi/4.0.1
export PYTHONPATH=/home/xz584/anaconda3/envs/gptune2/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$PWD/autotune/
export PYTHONPATH=$PYTHONPATH:$PWD/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$PWD/mpi4py/
export PYTHONPATH=$PYTHONPATH:$PWD/GPTune/
export PYTHONWARNINGS=ignore

CCC=mpicc
CCCPP=mpicxx
FTN=mpif90

#pip uninstall -r requirements.txt
#env CC=$CCC pip install --upgrade --user -r requirements.txt
env CC=$CCC pip install -r requirements.txt




wget http://www.netlib.org/scalapack/scalapack-2.1.0.tgz
tar -xf scalapack-2.1.0.tgz
cd scalapack-2.1.0
rm -rf build
mkdir -p build
cd build
cmake .. \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_C_COMPILER=$CCC \
    -DCMAKE_Fortran_COMPILER=$FTN \
    -DCMAKE_INSTALL_PREFIX=. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
    -DCMAKE_Fortran_FLAGS="-fopenmp" \
	-DBLAS_LIBRARIES="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;${MKLROOT}/lib/libmkl_def.so;${MKLROOT}/lib/libmkl_avx.so" \
	-DLAPACK_LIBRARIES="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;${MKLROOT}/lib/libmkl_def.so;${MKLROOT}/lib/libmkl_avx.so"
make -j8  
cd ../../
export SCALAPACK_LIB="$PWD/scalapack-2.1.0/build/lib/libscalapack.so" 



mkdir -p build
cd build
export CRAYPE_LINK_TYPE=dynamic
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="" \
	-DCMAKE_C_FLAGS="" \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;${MKLROOT}/lib/libmkl_def.so;${MKLROOT}/lib/libmkl_avx.so" \
	-DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;${MKLROOT}/lib/libmkl_def.so;${MKLROOT}/lib/libmkl_avx.so" \
	-DTPL_SCALAPACK_LIBRARIES="${SCALAPACK_LIB}"
make
cp lib_gptuneclcm.so ../.
cp pdqrdriver ../



cd ../examples/
rm -rf superlu_dist
git clone https://github.com/xiaoyeli/superlu_dist.git
cd superlu_dist

wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xf parmetis-4.0.3.tar.gz
cd parmetis-4.0.3/
mkdir -p install
make config shared=1 cc=$CCC cxx=$CCCPP prefix=$PWD/install
make install > make_parmetis_install.log 2>&1

cd ../
PARMETIS_INCLUDE_DIRS="$PWD/parmetis-4.0.3/metis/include;$PWD/parmetis-4.0.3/install/include"
PARMETIS_LIBRARIES=$PWD/parmetis-4.0.3/install/lib/libparmetis.so
mkdir -p build
cd build
rm -rf CMakeCache.txt
rm -rf DartConfiguration.tcl
rm -rf CTestTestfile.cmake
rm -rf cmake_install.cmake
rm -rf CMakeFiles
cmake .. \
	-DCMAKE_CXX_FLAGS="-Ofast -std=c++11 -DAdd_ -DRELEASE" \
	-DCMAKE_C_FLAGS="-std=c11 -DPRNTlevel=0 -DPROFlevel=0 -DDEBUGlevel=0" \
	-DBUILD_SHARED_LIBS=OFF \
	-DCMAKE_CXX_COMPILER=$CCCPP \
	-DCMAKE_C_COMPILER=$CCC \
	-DCMAKE_Fortran_COMPILER=$FTN \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
	-DTPL_BLAS_LIBRARIES="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;${MKLROOT}/lib/libmkl_def.so;${MKLROOT}/lib/libmkl_avx.so" \
	-DTPL_LAPACK_LIBRARIES="${MKLROOT}/lib/libmkl_gf_lp64.so;${MKLROOT}/lib/libmkl_gnu_thread.so;${MKLROOT}/lib/libmkl_core.so;${MKLROOT}/lib/libmkl_def.so;${MKLROOT}/lib/libmkl_avx.so" \
	-DTPL_PARMETIS_INCLUDE_DIRS=$PARMETIS_INCLUDE_DIRS \
	-DTPL_PARMETIS_LIBRARIES=$PARMETIS_LIBRARIES
make pddrive_spawn
make pzdrive_spawn


cd ../../
rm -rf hypre
git clone https://github.com/hypre-space/hypre.git
cd hypre/src/
# ./configure CC=$CCC CXX=$CCCPP FC=$FTN CFLAGS="-DTIMERUSEMPI -g -O0 -v -Q"
./configure CC=$CCC CXX=$CCCPP FC=$FTN CFLAGS="-DTIMERUSEMPI"
make
cp ../../hypre-driver/src/ij.c ./test/.
make test

# make CC=$CCC
cd ../../../
rm -rf mpi4py
git clone https://github.com/mpi4py/mpi4py.git
cd mpi4py/
python setup.py build --mpicc="$CCC -shared"
python setup.py install
# env CC=mpicc pip install --user -e .								  



cd ../
rm -rf scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize/
python setup.py build 
python setup.py install
# env CC=mpicc pip install --user -e .								  



cd ../
rm -rf autotune
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
env CC=$CCC pip install -e .


cp ../patches/opentuner/manipulator.py  /home/xz584/anaconda3/envs/gptune2/lib/python3.7/site-packages/opentuner/search/.
