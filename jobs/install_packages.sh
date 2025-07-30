#!/bin/bash
#SBATCH --job-name=install
#SBATCH --error=jobs/%j.txt
#SBATCH --output=jobs/%j.txt
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=23:59:00
cd $HOME/repos/tcv_eq
pip install -r requirements.txt