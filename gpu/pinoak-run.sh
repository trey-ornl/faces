#!/bin/bash
module load craype-accel-amd-gfx90a
module load rocm/5.4.3
module -t list
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
set -x
TASKS=$(( $1 * $2 * $3 ))
NODES=$(( ( TASKS - 1 ) / 8 + 1 ))
EXE=pinoak-faces
ldd "${EXE}" | grep -e mpi -e gtl -e amd
export OMP_NUM_THREADS=8
THREADS=$(( OMP_NUM_THREADS * 2 ))
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
#export MPICH_ENV_DISPLAY=1
echo "$1 $2 $3 15 14 13 12 10 10 100" | srun -K --exclusive -u -p bardpeak -t 1:00 -c ${THREADS} -N ${NODES} -n ${TASKS} ${EXE}
#echo "$1 $2 $3 15 14 13 12 1 1 10" | srun -K --exclusive -u -p bardpeak -t 1:00 -c ${THREADS} -N ${NODES} -n ${TASKS} ${EXE}
#echo "$1 $2 $3 105 104 103 12 3 1 10" | srun -K --exclusive -u -p bardpeak -t 5:00 -c ${THREADS} -N ${NODES} -n ${TASKS} ${EXE}
