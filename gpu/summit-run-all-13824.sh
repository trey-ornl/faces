#!/bin/bash
#BSUB -W 29
#BSUB -q debug
#BSUB -J summit-13824
#BSUB -nnodes 2304
#BSUB -q debug
module load nvphc
module -t list
set -x
X=24
Y=24
Z=24
TASKS=$(( X * Y * Z ))
for EXE in $(ls summit-faces*)
do
  echo "Running ${EXE}"
  echo "$X $Y $Z 95 94 93 12 3 1 10" | jsrun --smpiargs="-gpu" -n${TASKS} -r6 -c7 -EOMP_NUM_THREADS=7 -g1 -brs ${PWD}/${EXE}
    sleep 1
  echo "$X $Y $Z 15 14 13 12 10 10 100" | jsrun --smpiargs="-gpu" -n${TASKS} -r6 -c7 -EOMP_NUM_THREADS=7 -g1 -brs ${PWD}/${EXE}
  sleep 1
done
