#!/bin/bash
#SBATCH --job-name=cmae-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --constraint="nomig&gpu40"
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/output.txt
#SBATCH --error=logs/error.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=ra0438@princeton.edu

cd ~/chime-catalog-two-analysis

module load anaconda3/2025.12
conda activate frb_env

python plot_cls_map.py