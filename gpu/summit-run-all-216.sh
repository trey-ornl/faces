#!/bin/bash
#BSUB -W 14
##BSUB -q debug
#BSUB -J summit-216
#BSUB -nnodes 36
module load nvphc
module -t list
set -x
X=6
Y=6
Z=6
TASKS=$(( X * Y * Z ))
for EXE in $(ls summit-faces*)
do
  echo "Running ${EXE}"
  sleep 1
  echo "$X $Y $Z 15 14 13 12 10 10 100" | jsrun --smpiargs="-gpu" -n${TASKS} -r6 -c7 -EOMP_NUM_THREADS=7 -g1 -brs ${PWD}/${EXE}
  sleep 1
  echo "$X $Y $Z 105 104 103 12 3 1 10" | jsrun --smpiargs="-gpu" -n${TASKS} -r6 -c7 -EOMP_NUM_THREADS=7 -g1 -brs ${PWD}/${EXE}
done
