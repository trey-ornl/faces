#!/bin/bash
module load nvhpc
module -t list
set -x
export CXX=mpiCC
export CXXFLAGS="-g -O3 -std=c++17 -cuda -mp"
export LD="${CXX}"
export LDFLAGS="${CXXFLAGS}"
export LIBS=''

MAXTHREADS=${1:-32}
build () {
  DIR=$1
  for FILE in $(ls ${DIR}/)
  do
    ln -f -s ${DIR}/${FILE} .
  done
  export EXE=summit-faces-t${MAXTHREADS}-$1$2
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
build v-6-messages-overlap-fusable -XYZ

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS}"
build v-26-messages
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND"
build v-26-messages -send
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_INNER"
build v-26-messages -inner
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_RECV"
build v-26-messages -recv
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND -DFUSE_INNER -DFUSE_RECV"
build v-26-messages -fused
export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND -DFUSE_INNER"
build v-26-messages -send-inner

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND -DFUSE_INNER -DFUSE_RECV"
build v-26-messages-device-sync -fused

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_INNER"
build v-26-messages-events -inner

export CPPLFAGS="-DFUSE_SEND -DFUSE_INNER"
build v-26-messages-piplined-recvs -send-inner

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND -DFUSE_INNER -DFUSE_RECV"
build v-26-messages-single-stream -fused

export CPPFLAGS="-DMAXTHREADS=${MAXTHREADS} -DFUSE_SEND -DFUSE_INNER -DFUSE_RECV"
build v-26-messages-single-stream-event -fused

