#!/bin/bash
source frontier-env
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
export OMP_NUM_THREADS=7
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1

for EXE in $(ls frontier-faces*)
do
  echo "Running ${EXE}"
  unset MPICH_RANK_REORDER_METHOD
  sleep 1
  echo "1 1 1 15 14 13 12 10 10 100" | ./${EXE}
  sleep 1
  echo "1 1 1 95 94 93 12 3 1 10" | ./${EXE}
done
