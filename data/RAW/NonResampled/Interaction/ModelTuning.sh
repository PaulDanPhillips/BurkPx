#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=BurkPx
##SBATCH --output=
##SBATCH --array=1-4
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=5000

#name1=$(sed -n "$SLURM_ARRAY_TASK_ID"p FileLST.txt)
#name2=$(sed -n "$SLURM_ARRAY_TASK_ID"p Responses.txt)
#redux=$(echo $name2 | cut -f 1 -d '.')

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python ModelTuning.py -p Predictor_IgGMall_PosNeg_interact.txt -r Response.txt -k 75 -a roc_auc -o Results
