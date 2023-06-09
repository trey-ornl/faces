#!/bin/bash
module load craype-accel-amd-gfx90a
module load rocm
module -t list
set -x
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export CXX=hipcc
export CXXFLAGS="-g -O3 -std=c++17 --offload-arch=gfx90a -Wall -I${CRAY_MPICH_DIR}/include"
export LD=CC
export LDFLAGS="-g -O3 -std=c++17 -Wall -L${ROCM_PATH}/lib"
export LIBS='-lamdhip64'
export EXE=frontier-faces
#make clean
#make -j
make && \
ldd "${EXE}" | grep -e mpi -e gtl
