# Faces
Faces is a microbenchmark for testing programming strategies for nearest-neighbor communication with GPU-aware MPI.

This repo contains the Faces versions used for the paper, "Performance Portability of Programming Strategies for Nearest-Neighbor Communication with GPU-Aware MPI," presented at the [2023 International Workshop on Performance, Portability & Productivity in HPC](https://p3hpc.org/workshop/2023/).

Here is a summary of the commands used to generate the results in that paper.

## Installation
```
git clone git@github.com:twhite-cray/faces.git
cd gpu
```
## Experiment Workflow
### Single-GPU Experiments
```
# Frontier
rm -f frontier-faces-*
./frontier-make-fusion.sh
./frontier-make-fusion.sh 256
./frontier-make-fusion.sh 1024
sbatch -N1 --exclusive -t 30 -A ${ACCT} ./frontier-run-all-1.sh
```
```
# Summit
rm -f summit-faces-*
./summit-make-fusion.sh
./summit-make-fusion.sh 256
./summit-make-fusion.sh 1024
bsub -P ${ACCT} summit-run-all-1.sh
```
### Scaling Experiments
```
# Frontier
rm -f frontier-faces-*
./frontier-make-scale.sh
sbatch -n216 -N27 --exclusive -t 30 -A ${ACCT} ./frontier-run-all.sh 6 6 6
sbatch -n1728 -N216 --exclusive -t 30 -A ${ACCT} ./frontier-run-all.sh 12 12 12
sbatch -n13824 -N1728 --exclusive -t 30 -A ${ACCT} ./frontier-run-all.sh 24 24 24
```
```
# Summit
rm -f summit-faces-*
./summit-make-scale.sh
bsub -P ${ACCT} summit-run-all-216.sh
bsub -P ${ACCT} summit-run-all-1728.sh
bsub -P ${ACCT} summit-run-all-13824.sh
```
Each build should take a few minutes. Each batch job should take tens of minutes to run.

## Evaluation and Expected Results
Run the following command on the output file from each batch script.
```
grep -e ^Run -e ^time -e correct <file>
```
The `Running` lines show which executables ran. Confirm that all `tasks passed correctness checks`. Use the `max` value from the `time` line for each run. Compare times from runs within a job to assess performance trade-offs.

## Experiment Customization

See the `*-make-*.sh` scripts to modify builds. The scripts set standard `Makefile` variables. Modify `CPPFLAGS` to set max threads and to control kernel fusion. Modify or add calls to the `build` function defined in each script to change what executables get built.

See the `*-run-all*.sh` scripts to modify runs. The Faces executables read the following arguments in order from `stdin`: *l<sub>x</sub> l<sub>y</sub> l<sub>z</sub> m<sub>x</sub> m<sub>y</sub> m<sub>z</sub> n n<sub>k</sub> n<sub>j</sub> n<sub>i</sub>*, where the MPI grid is *l<sub>x</sub>* x *l<sub>y</sub>* x *l<sub>z</sub>*, the local spectral-element grid is *m<sub>x</sub>* x *m<sub>y</sub>* x *m<sub>z</sub>*, each spectral element has *n*<sup>3</sup> points, and the nested run loops have bounds of *n<sub>k</sub>*, *n<sub>j</sub>*, and *n<sub>i</sub>*.

Use different arguments to `frontier-run-all.sh` to use a different MPI grid on Frontier. Modify one of the `summit-run-all-*.sh` scripts to use a different MPI grid on Summit. Each script will run whatever (appropriately named) Faces executables it finds in the same directory.

