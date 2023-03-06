#!/bin/bash
#SBATCH -c 9
#SBATCH --mem 128GB
#SBATCH --time 48:00:00
#SBATCH --gres gpu:2
#SBATCH --job-name test_ss_on_meccano
#SBATCH --output sbatch_logs/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs3639@nyu.edu


./sing << EOF

python runs/run.py --cfg configs/projects/hyrsm/meccano/HyRSM_Epic_5shot_v1_test.yaml \
    OUTPUT_DIR output/test_ss_on_meccano \
    TRAIN.CHECKPOINT_FILE_PATH output/HyRSM_SSv2_Full_5shot/checkpoints/checkpoint_epoch_00012.pyth \
    TRAIN.NUM_CLASS 64

EOF
