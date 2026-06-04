#!/bin/bash
#SBATCH --job-name=cmae-optimize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint="intel&gpu40"
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ra0438@princeton.edu

cd ~/chime-catalog-two-analysis/optimize

module load anaconda3/2025.12
conda activate frb_env

python cmae-optimize.py