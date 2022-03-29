#!/usr/bin/env bash
source activate py37_tf115

export PYTHONPATH=./src  # the path of the src folder
DATA_HOME=./Koubei  # the path of the data folder

# fin  -> the path of user-item-timestamp triplet data
# fout -> the output folder of TFRECORD files
# seqslen -> the maximum length of the sequence
python data/learningtorank.py \
  --seqslen=30 --fin=${DATA_HOME}/user_train.csv \
  --fout=${DATA_HOME}

# train -> the TFRECORD files
# test  -> the TFRECORD files
# num_items -> need to specify the number of items
# seqslen -> the maximum length of the sequence
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --train="${DATA_HOME}/train???.tfrec" \
    --test=${DATA_HOME}/test.tfrec \
    --learning_rate=1e-4 --num_units=512 --dropout_rate=0. \
    --num_train_steps=0 --num_warmup_steps=0 \
    --num_items=10214 --seqslen=30 --model=S2PNM --batch_size=512 --num_epochs=100

conda deactivate