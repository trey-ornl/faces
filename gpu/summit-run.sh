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
#echo "3 2 1 15 14 13 12 10 10 100" | jsrun --smpiargs="-gpu" -n6 -r6 -c7 -EOMP_NUM_THREADS=7 -g1 -brs ${EXE}
echo "3 2 1 105 104 103 12 1 1 10" | jsrun --smpiargs="-gpu" -n6 -r6 -c7 -EOMP_NUM_THREADS=7 -g1 -brs ${EXE}
