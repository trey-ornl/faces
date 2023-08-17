#!/bin/bash
module load craype-accel-amd-gfx90a
module load rocm
module -t list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS - 1 ) / 8 + 1 ))
export OMP_NUM_THREADS=7
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
export MPICH_RANK_REORDER_FILE="MPICH_RANK_ORDER.${SLURM_JOB_ID}"
if [ ${NODES} -gt 1 ]
then
  grid_order -C -c 2,2,2 -g $1,$2,$3 | tee ${MPICH_RANK_REORDER_FILE}
fi

date
for EXE in $(ls frontier-faces*)
do
  echo "Running ${EXE}"
  unset MPICH_RANK_REORDER_METHOD
  sleep 1
  echo "$1 $2 $3 15 14 13 12 10 10 100" | srun --exclusive -K -u -c ${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest -N ${NODES} -n ${TASKS} ${EXE}
  sleep 1
  echo "$1 $2 $3 105 104 103 12 3 1 10" | srun --exclusive -K -u -c ${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest -N ${NODES} -n ${TASKS} ${EXE}
  if [ ${NODES} -gt 1 ]
  then
    export MPICH_RANK_REORDER_METHOD=3
    sleep 1
    echo "$1 $2 $3 15 14 13 12 10 10 100" | srun --exclusive -K -u -c ${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest -N ${NODES} -n ${TASKS} ${EXE}
    sleep 1
    echo "$1 $2 $3 105 104 103 12 3 1 10" | srun --exclusive -K -u -c ${OMP_NUM_THREADS} --gpus-per-task=1 --gpu-bind=closest -N ${NODES} -n ${TASKS} ${EXE}
  fi
done
date
