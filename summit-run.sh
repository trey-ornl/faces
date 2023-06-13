#!/bin/bash
#BSUB -W 5 
#BSUB -P VEN114
#BSUB -J summit-faces
#BSUB -nnodes 1
module load nvphc
module -t list
set -x
EXE=${PWD}/summit-faces
ldd "${EXE}"
echo "3 2 1 15 14 13 12 10 10 100" | jsrun --smpiargs="-gpu" -n6 -g1 ${EXE}
env
