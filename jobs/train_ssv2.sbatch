#!/bin/bash
#SBATCH -c 9
#SBATCH --mem 128GB
#SBATCH --time 48:00:00
#SBATCH --gres gpu:2
#SBATCH --job-name ssv2_hyrsm_train
#SBATCH --output sbatch_logs/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs3639@nyu.edu


./sing << EOF

python runs/run.py --cfg configs/projects/hyrsm/ssv2_full/HyRSM_SSv2_Full_5shot_v1.yaml

EOF
