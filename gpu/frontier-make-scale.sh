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

build () {
  DIR=$1
  for FILE in $(ls ${DIR}/)
  do
    ln -f -s ${DIR}/${FILE} .
  done
  export EXE=frontier-faces-t$1-$2$3
  make clean
  make -j
}

rm -f frontier-faces*

MAXTHREADS=256
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS}"
build ${MAXTHREADS} v-6-messages-basic
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_Y"
build ${MAXTHREADS} v-6-messages-overlap-fusable -Y

MAXTHREADS=64
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_INNER"
build ${MAXTHREADS} v-26-messages -inner
build ${MAXTHREADS} v-26-messages-events -inner
build ${MAXTHREADS} v-26-messages-pipelined-recvs -inner
build ${MAXTHREADS} v-26-messages-single-stream -inner
build ${MAXTHREADS} v-26-messages-single-stream-event -inner
build ${MAXTHREADS} v-26-messages-streams -inner

