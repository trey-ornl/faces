#!/bin/bash
module load craype-accel-amd-gfx90a
module load rocm/5.4.3
module -t list
set -x
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export CXX=hipcc
export CXXFLAGS="-DFUSE_SEND -DFUSE_INNER -g --offload-arch=gfx90a -O3 -std=c++17 -Wall -fopenmp -I${CRAY_MPICH_DIR}/include"
export LD=CC
export LDFLAGS="-g -O3 -std=c++17 -Wall -fopenmp"
export LIBS="-L${ROCM_PATH}/lib -lamdhip64"
export EXE=pinoak-faces
make clean
make -j
ldd "${EXE}" | grep -e mpi -e gtl -e amd -e mp
