#!/bin/bash
#SBATCH -c 9
#SBATCH --mem 128GB
#SBATCH --time 48:00:00
#SBATCH --gres gpu:2
#SBATCH --job-name test_ek_on_ek_noaction
#SBATCH --output sbatch_logs/%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bs3639@nyu.edu


./sing << EOF

python runs/run.py --cfg configs/projects/hyrsm/epic_kitchens/HyRSM_Epic_5shot_v1_test.yaml  \
    OUTPUT_DIR output/test_ek_on_ek_noaction \
    TRAIN.CHECKPOINT_FILE_PATH output/HyRSM_Epic_5shot/checkpoints/checkpoint_epoch_00005.pyth \
    DATA.UNSPECIFIED true

EOF
