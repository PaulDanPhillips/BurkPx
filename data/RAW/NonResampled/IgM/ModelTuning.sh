#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=BurkPx
##SBATCH --output=
#SBATCH --array=1-4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=5000

name1=$(sed -n "$SLURM_ARRAY_TASK_ID"p FileLST.txt)
#name2=$(sed -n "$SLURM_ARRAY_TASK_ID"p Responses.txt)
redux=$(echo $name1 | cut -f 1 -d '_')

echo $name1
echo $redux

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python ModelTuning.py -f $name1 -k 250 -a balanced_accuracy -o Results -s $redux 
