#!/bin/bash
module load nvhpc
module -t list
set -x
export CXX=mpiCC
export CXXFLAGS="-g -O3 -std=c++17 -cuda -mp"
export LD="${CXX}"
export LDFLAGS="${CXXFLAGS}"
export LIBS=''
export EXE=summit-faces
make clean
make -j && \
ldd "${EXE}" | grep -e mpi -e cuda
