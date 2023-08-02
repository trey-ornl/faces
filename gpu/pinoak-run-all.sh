#!/bin/bash
module load craype-accel-amd-gfx90a
module load rocm/5.4.3
module -t list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS - 1 ) / 8 + 1 ))
export OMP_NUM_THREADS=8
THREADS=$(( 2 * OMP_NUM_THREADS ))
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_VERSION_DISPLAY=1
grid_order -R -c 2,2,2 -g $1,$2,$3 | tee MPICH_RANK_ORDER

for EXE in $(ls pinoak-faces-v*)
do
  echo "Running ${EXE}"
  unset MPICH_RANK_REORDER_METHOD
  sleep 1
  echo "$1 $2 $3 15 14 13 12 10 10 100" | srun -K --exclusive -u -p bardpeak -c ${THREADS} -N ${NODES} -n ${TASKS} bash -c "GPU_MAP=(4 5 2 3 6 7 0 1); GPU_ID=\$((8 * SLURM_LOCALID * SLURM_NNODES / SLURM_NTASKS)); env HIP_VISIBLE_DEVICES=\${GPU_MAP[GPU_ID]} ./${EXE}"
  sleep 1
  echo "$1 $2 $3 105 104 103 12 3 1 10" | srun -K --exclusive -u -p bardpeak -c ${THREADS} -N ${NODES} -n ${TASKS} bash -c "GPU_MAP=(4 5 2 3 6 7 0 1); GPU_ID=\$((8 * SLURM_LOCALID * SLURM_NNODES / SLURM_NTASKS)); env HIP_VISIBLE_DEVICES=\${GPU_MAP[GPU_ID]} ./${EXE}"
  export MPICH_RANK_REORDER_METHOD=3
  sleep 1
  echo "$1 $2 $3 15 14 13 12 10 10 100" | srun -K --exclusive -u -p bardpeak -c ${THREADS} -N ${NODES} -n ${TASKS} bash -c "GPU_MAP=(4 5 2 3 6 7 0 1); GPU_ID=\$((8 * SLURM_LOCALID * SLURM_NNODES / SLURM_NTASKS)); env HIP_VISIBLE_DEVICES=\${GPU_MAP[GPU_ID]} ./${EXE}"
  sleep 1
  echo "$1 $2 $3 105 104 103 12 3 1 10" | srun -K --exclusive -u -p bardpeak -c ${THREADS} -N ${NODES} -n ${TASKS} bash -c "GPU_MAP=(4 5 2 3 6 7 0 1); GPU_ID=\$((8 * SLURM_LOCALID * SLURM_NNODES / SLURM_NTASKS)); env HIP_VISIBLE_DEVICES=\${GPU_MAP[GPU_ID]} ./${EXE}"
done
