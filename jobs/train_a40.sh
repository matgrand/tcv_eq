#!/bin/bash
#SBATCH --job-name=t1a40
#SBATCH --error=jobs/%j.txt
#SBATCH --output=jobs/%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:a40:1
cd $HOME/repos/tcv_eq
echo "running job $SLURM_JOB_ID"
echo "Latest commit:"
git log -1 --pretty=format:"[%h] %s"
echo
echo "Running training script A40..."
srun jupyter nbconvert train.ipynb --to python && python train.py && rm -rf train.py
echo "Latest commit:"
git log -1 --pretty=format:"[%h] %s"
echo "Training script completed."