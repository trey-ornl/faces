#!/bin/bash
module load nvhpc
module -t list
set -x
export CXX=mpiCC
export CXXFLAGS="-g -O3 -std=c++17 -cuda -mp"
export LD="${CXX}"
export LDFLAGS="${CXXFLAGS}"
export LIBS=''

build () {
  DIR=$2
  for FILE in $(ls ${DIR}/)
  do
    ln -f -s ${DIR}/${FILE} .
  done
  export EXE=summit-faces-t$1-$2$3
  make clean
  make -j
}

rm -f summit-faces*

MAXTHREADS=32
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS}"
build ${MAXTHREADS} v-6-messages-basic
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_X -DFUSE_Y -DFUSE_Z"
build ${MAXTHREADS} v-6-messages-overlap-fusable -XYZ

MAXTHREADS=32
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_INNER -DFUSE_RECV -DFUSE_SEND"
build ${MAXTHREADS} v-26-messages -fused
build ${MAXTHREADS} v-26-messages-events -fused
build ${MAXTHREADS} v-26-messages-pipelined-recvs -fused
build ${MAXTHREADS} v-26-messages-single-stream -fused
build ${MAXTHREADS} v-26-messages-single-stream-event -fused
build ${MAXTHREADS} v-26-messages-streams -fused

