#!/bin/bash
#SBATCH --chdir /scratch/izar/rajic/vo
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 92160
#SBATCH --time 24:00:00
#SBATCH --account cs503
#SBATCH --reservation courses
#SBATCH --gres=gpu:1
#SBATCH -q gpu
#SBATCH -o ./slurm_logs/slurm-sbatch_8-25-%j.out

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Modules
module purge
module load gcc/8.4.0-cuda
module load cuda/10.1
module load parallel

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ruslan

# Run
date
printf "Run configured and environment setup. Gonna run now.\n\n"
python -m pointnav_vo.run --task-type rl --noise 1 --exp-config configs/rl/ddppo_pointnav.yaml --run-type eval --n-gpu 1 --cur-time 123 --video_log_interval 200 --challenge_config_file config_files/challenge_pointnav2021.local.rgbd.GPU.yaml --agent_name vo --dataset_split val --seed 72 -vc Speckle_Noise -vs 3 --pyrobot_noise_multiplier 0.0
echo FINISHED at $(date)
