#!/bin/bash

singularity exec --nv \
    --no-home \
    --bind $PWD:$PWD \
    /da/CADD/singularity/images/chemprop-1.6.1-cu118.sif \
    chemprop_hyperopt \
        --data_path input_chemprop.csv \
        --dataset_type regression \
        --target_columns 'A_pIC50' \
        --smiles_columns "Structure" \
        --split_type random \
        --split_sizes 0.8 0.1 0.1 \
        --num_iters 200 \
        --num_folds 5 \
        --save_dir $SAVEDIR \
        --save_preds \
        --epochs 50 \
        --batch_size 32 \
        --extra_metrics r2 mae \
        --ensemble_size 1 \
        --save_dir $PWD/hyperopt \
        --config_save_path $PWD/hyperopt/best_params.json \

