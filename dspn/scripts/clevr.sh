#!/bin/bash


DATASET=$1
NUM=$2
USE_BASELINE=$3

if [ ! -z "$USE_BASELINE" ]
then
    PREFIX="base"
    ARGS="--baseline --decoder MLPDecoder"
else
    PREFIX="dspn"
    ARGS="--decoder DSPN"
fi

set -x

python train.py --show --loss hungarian --encoder RNFSEncoder --dim 512 --dataset $DATASET --epochs 100 --latent 512 --supervised --name $PREFIX-$DATASET-$NUM --iters 10 --lr 0.0003 --huber-repr 0.1 --mask-feature $ARGS
