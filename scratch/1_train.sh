#!/bin/bash

singularity exec --nv \
    --no-home \
    --bind $PWD:$PWD \
    /da/CADD/singularity/images/chemprop-1.6.1-cu118.sif \
    chemprop_train \
        --data_path input_chemprop.csv \
        --dataset_type regression \
        --target_columns 'A_pIC50' \
                         'B_pIC50' \
        --smiles_columns "Structure" \
        --split_type random \
        --num_folds 5 \
        --save_dir $SAVEDIR \
        --save_preds \
        --epochs 50 \
        --batch_size 32 \
        --show_individual_scores \
        --save_smiles_splits \
        --metric "r2" \
        --extra_metrics "rmse" "mae" \
        --ensemble_size 1 \

