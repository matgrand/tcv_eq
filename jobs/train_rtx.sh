#!/bin/bash
#SBATCH --job-name=t1rtx
#SBATCH --error=jobs/%j.txt
#SBATCH --output=jobs/%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=19:59:00
#SBATCH --gres=gpu:rtx:1
cd $HOME/repos/tcv_eq
echo "running job $SLURM_JOB_ID"
srun jupyter nbconvert train.ipynb --to python && python train.py && rm -rf train.py