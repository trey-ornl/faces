#!/bin/bash
source frontier-env
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS - 1 ) / 8 + 1 ))
EXE=frontier-faces
ldd "${EXE}" | grep -e mpi -e gtl -e amd
export OMP_NUM_THREADS=7
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_ENV_DISPLAY=1
echo "$1 $2 $3 15 14 13 12 10 10 100" | srun -q debug --exclusive -u -t 5:00 -c ${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest -N ${NODES} -n ${TASKS} ${EXE}
echo "$1 $2 $3 95 94 93 12 1 1 10" | srun -q debug --exclusive -u -t 5:00 -c ${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest -N ${NODES} -n ${TASKS} ${EXE}
