#!/bin/bash
module load craype-accel-amd-gfx90a
module load rocm
module -t list
set -x
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export CXX=hipcc
export CXXFLAGS="-g --offload-arch=gfx90a -O3 -std=c++17 -Wall -fopenmp -I${CRAY_MPICH_DIR}/include"
export LD=hipcc
export LDFLAGS="-g -O3 -std=c++17 -Wall -fopenmp"
export LIBS="-L${MPICH_DIR}/lib -lmpi ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -L${ROCM_PATH}/lib -lamdhip64"
export EXE=frontier-faces
make clean
make -j
ldd "${EXE}" | grep -e mpi -e gtl -e amd -e mp
