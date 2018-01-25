#!/usr/bin/bash
FRA_WD="wd"
FRA_MODEL_NAME="test"

# check seted date, if cat be used for load pretrained models
DATE="test"
# using directory
FRA_MODELDIR=${FRA_WD}/models/$DATE
# make using directory
mkdir -p $FRA_MODELDIR
# save short hash of git commit




 time python train.py     \
    --work_dir=$FRA_WD \
    --model_name=$FRA_MODEL_NAME \
    --save_dir=$FRA_MODELDIR \
    --save_dir=$FRA_MODELDIR \
    --min_sentence_length=3 \
    --max_sentence_length=80 \
    --attn_model='general' \
    --hidden_size=500 \
    --n_layers=2 \
    --dropout=0.1 \
    --batch_size=100 \
    --clip=50.0 \
    --learning_rate=0.0001 \
    --decoder_learning_ratio=5.0 \
    --n_epochs=10 \
    --plot_every=2 \
    --print_every=2
