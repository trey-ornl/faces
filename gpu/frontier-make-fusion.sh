#!/bin/bash
source frontier-env
module -t list
set -x
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export CXX=hipcc
export CXXFLAGS="-g --offload-arch=gfx90a -O3 -std=c++17 -Wall -fopenmp -I${CRAY_MPICH_DIR}/include"
export LD=hipcc
export LDFLAGS="-g -O3 -std=c++17 -Wall -fopenmp"
export LIBS="-L${MPICH_DIR}/lib -lmpi ${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem ${PE_MPICH_GTL_DIR_amd_gfx90a} ${PE_MPICH_GTL_LIBS_amd_gfx90a} -L${ROCM_PATH}/lib -lamdhip64"

MAXTHREADS=${1:-64}
build () {
  DIR=$1
  for FILE in $(ls ${DIR}/)
  do
    ln -f -s ${DIR}/${FILE} .
  done
  export EXE=frontier-faces-t${MAXTHREADS}-$1$2
  make clean
  make -j
}

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS}"
build v-6-messages-basic

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS}"
build v-6-messages-overlap-fusable
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_X"
build v-6-messages-overlap-fusable -X
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_Y"
build v-6-messages-overlap-fusable -Y
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_Z"
build v-6-messages-overlap-fusable -Z
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_X -DFUSE_Y -DFUSE_Z"
build v-6-messages-overlap-fusable -all

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS}"
build v-26-messages
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_INNER"
build v-26-messages -inner
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_RECV"
build v-26-messages -recv
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND"
build v-26-messages -send
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND -DFUSE_INNER -DFUSE_RECV"
build v-26-messages -all
