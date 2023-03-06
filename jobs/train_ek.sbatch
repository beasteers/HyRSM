#!/bin/bash
#SBATCH -c 9
#SBATCH --mem 128GB
#SBATCH --time 48:00:00
#SBATCH --gres gpu:2
#SBATCH --job-name hyrsm_train2
#SBATCH --output sbatch_logs/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs3639@nyu.edu


./sing << EOF

python runs/run.py --cfg configs/projects/hyrsm/epic_kitchens/HyRSM_Epic_5shot_v1.yaml 

EOF
