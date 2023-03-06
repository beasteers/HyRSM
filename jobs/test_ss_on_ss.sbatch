#!/bin/bash
#SBATCH -c 9
#SBATCH --mem 128GB
#SBATCH --time 48:00:00
#SBATCH --gres gpu:2
#SBATCH --job-name test_ss_on_ss
#SBATCH --output sbatch_logs/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs3639@nyu.edu


./sing << EOF

python runs/run.py --cfg configs/projects/hyrsm/ssv2_full/HyRSM_SSv2_Full_5shot_v1_test.yaml \
    OUTPUT_DIR output/test_ss_on_ss \
    TRAIN.CHECKPOINT_FILE_PATH output/HyRSM_SSv2_Full_5shot/checkpoints/checkpoint_epoch_00012.pyth

EOF
