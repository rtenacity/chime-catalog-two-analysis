# Conda Setup

module load anaconda3/2025.12
conda activate frb_env

# Queue job
sbatch <job_script>.slurm (replace with actual script name)
squeue -u $USER -l (to check job status with reasoning)
scancel <job_id> (to cancel a job with reasoning)

[SLURM documentation](https://researchcomputing.princeton.edu/support/knowledge-base/slurm)

