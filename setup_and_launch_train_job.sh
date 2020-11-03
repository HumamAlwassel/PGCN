#!/bin/bash --login

NUM_RUNS=6
START_RUN_ID=0
BATCH_SIZE=32
LR=0.01
FEAT_DIM=512

OUTPUT_ROOT=/ibex/scratch/alwassha/pytorch-experiments/pgcn/e2e-video_features_thumos14/features_stride_1_interpolated

for FEATURE_TYPE in \
r2plus1d-18_features_one-head_0.001-0.001-0.001-0.001-0.002_model_3-interpolated \
r2plus1d-18_features_one-head_fc-only-0.002_model_0-interpolated \
r2plus1d-18_features_two-heads-A-noA-1.0-1.0_0.001-0.001-0.001-0.001-0.002_model_4-interpolated \
r2plus1d-18_features_two-heads-A-noA-with-global-max-features-1.0-1.0_0.001-0.001-0.001-0.001-0.002_model_6-interpolated \
r2plus1d-34_features_one-head_0.00001-0.00001-0.00001-0.00001-0.002_model_3-interpolated \
r2plus1d-34_features_one-head_fc-only-0.002_model_0-interpolated \
r2plus1d-34_features_two-heads-A-noA-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.002_model_1-interpolated \
r2plus1d-34_features_two-heads-A-noA-with-global-avg-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_7-interpolated\
r2plus1d-34_features_two-heads-A-noA-with-global-max-features-1.0-1.0_0.0001-0.0001-0.0001-0.0001-0.004_model_4-interpolated 
do

    FEATURE_PATH=/ibex/scratch/alwassha/e2e-video_features/thumos14/pgcn_features_stride_1_interpolated/${FEATURE_TYPE}.h5

    for i in $( seq $START_RUN_ID $(( START_RUN_ID + NUM_RUNS - 1 )) )
    do
        RUN_ID=run_${i}
        OUTPUT_PATH=${OUTPUT_ROOT}/${FEATURE_TYPE}/LR_${LR}-SF_${SKIP_VIDEOFRAMES}-TS_${TEMPORAL_SCALE}-with-cls-data/${RUN_ID}
        JOB_NAME=PGCN-TH14-${FEATURE_TYPE}-${LR}-${RUN_ID}

        mkdir -p $OUTPUT_PATH
        echo $JOB_NAME

        sbatch --gres=gpu:1 --job-name=${JOB_NAME} \
        --export=ALL,FEATURE_TYPE=$FEATURE_TYPE,FEATURE_PATH=$FEATURE_PATH,FEAT_DIM=$FEAT_DIM,BATCH_SIZE=$BATCH_SIZE,LR=$LR,OUTPUT_PATH=$OUTPUT_PATH,JOB_NAME=$JOB_NAME \
        slurm_train_pgcn.sh
    done

done
done
