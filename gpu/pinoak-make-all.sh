#!/bin/bash
module load craype-accel-amd-gfx90a
module load rocm/5.4.3
module -t list
set -x
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export CXX=hipcc
export CXXFLAGS="-g --offload-arch=gfx90a -O3 -std=c++17 -Wall -fopenmp -I${CRAY_MPICH_DIR}/include"
export LD=CC
export LDFLAGS="-g -O3 -std=c++17 -Wall -fopenmp"
export LIBS="-L${ROCM_PATH}/lib -lamdhip64"

build () {
  DIR=$1
  for FILE in $(ls ${DIR}/)
  do
    ln -f -s ${DIR}/${FILE} .
  done
  export EXE=pinoak-faces-$1$2
  make clean
  make -j
}

rm -f pinoak-faces-*

unset CPPFLAGS
build v-6-messages-basic

unset CPPFLAGS
build v-6-messages-overlap-fusable
export CPPFLAGS='-DFUSE_X'
build v-6-messages-overlap-fusable -X
export CPPFLAGS='-DFUSE_Y'
build v-6-messages-overlap-fusable -Y
export CPPFLAGS='-DFUSE_Z'
build v-6-messages-overlap-fusable -Z
export CPPFLAGS='-DFUSE_X -DFUSE_Y -DFUSE_Z'
build v-6-messages-overlap-fusable -XYZ

unset CPPFLAGS
build v-26-messages
export CPPFLAGS='-DFUSE_SEND'
build v-26-messages -send
export CPPFLAGS='-DFUSE_INNER'
build v-26-messages -inner
export CPPFLAGS='-DFUSE_RECV'
build v-26-messages -recv
export CPPFLAGS='-DFUSE_SEND -DFUSE_INNER -DFUSE_RECV'
build v-26-messages -fused
export CPPFLAGS='-DFUSE_SEND -DFUSE_INNER'
build v-26-messages -send-inner

export CPPFLAGS='-DFUSE_SEND -DFUSE_INNER -DFUSE_RECV'
build v-26-messages-device-sync -fused

export CPPFLAGS='-DFUSE_INNER'
build v-26-messages-events -inner

export CPPLFAGS='-DFUSE_SEND -DFUSE_INNER'
build v-26-messages-piplined-recvs -send-inner

export CPPFLAGS='-DFUSE_SEND -DFUSE_INNER -DFUSE_RECV'
build v-26-messages-single-stream -fused

export CPPFLAGS='-DFUSE_SEND -DFUSE_INNER -DFUSE_RECV'
build v-26-messages-single-stream-event -fused

