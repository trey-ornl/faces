#!/bin/bash
#BSUB -W 50
##BSUB -q debug
#BSUB -J summit-1
#BSUB -nnodes 1
module load nvphc
module -t list
set -x
X=1
Y=1
Z=1
TASKS=$(( X * Y * Z ))
date
for EXE in $(ls summit-faces*)
do
  echo "Running ${EXE}"
  sleep 1
  echo "$X $Y $Z 15 14 13 12 10 10 100" | jsrun --smpiargs="-gpu" -n${TASKS} -c7 -EOMP_NUM_THREADS=7 -g1 ${PWD}/${EXE}
  sleep 1
  echo "$X $Y $Z 105 104 103 12 3 1 10" | jsrun --smpiargs="-gpu" -n${TASKS} -c7 -EOMP_NUM_THREADS=7 -g1  ${PWD}/${EXE}
done
date
