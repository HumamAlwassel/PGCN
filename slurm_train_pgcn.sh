#!/bin/bash --login
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%A-%3a.out
#SBATCH --error=logs/%x-%A-%3a.err
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=40G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=humam.alwassel@kaust.edu.sa
#SBATCH -A conf-gpu-2020.11.23
#SBATCH -x gpu213-10,gpu210-06,gpu510-12,gpu510-17,gpu214-18,gpu213-14

set -ex

hostname
nvidia-smi
env

python pgcn_train.py thumos14 \
--workers 6 \
--batch-size $BATCH_SIZE \
--lr $LR \
--feat_dim $FEAT_DIM \
--train_ft_path ${FEATURE_PATH} \
--test_ft_path ${FEATURE_PATH} \
--snapshot_pre ${OUTPUT_PATH} 

python pgcn_test.py thumos14 ${OUTPUT_PATH}/PGCN_thumos14_model_best.pth.tar ${OUTPUT_PATH}/detection_results \
--workers 6 \
--feat_dim $FEAT_DIM \
--train_ft_path ${FEATURE_PATH} \
--test_ft_path ${FEATURE_PATH} 

python eval_detection_results.py thumos14 ${OUTPUT_PATH}/detection_results  \
--nms_threshold 0.35 \
--feat_dim $FEAT_DIM \
--train_ft_path ${FEATURE_PATH} \
--test_ft_path ${FEATURE_PATH} 
