#!/bin/bash

MODEL=$1
LIMIT=$2
USE_BASELINE=$3
THRESHOLDS="inf 1 0.5 0.25 0.125"
ITERS="10 20 30"

if [ ! -z "$USE_BASELINE" ]
then
    ARGS="--baseline --decoder RNNDecoder"
    ITERS="10"
fi

for iter in $ITERS; do
    OUTPATH="out/clevr-state/$MODEL-$iter"
    python train.py --show --loss hungarian --encoder RNFSEncoder --dim 512 --dataset clevr-state --epochs 100 --latent 512 --supervised --resume logs/$MODEL --iters $iter --name test --eval-only --export-dir $OUTPATH --full-eval $ARGS --mask-feature

    for t in $THRESHOLDS; do
        scripts/eval-state.sh $OUTPATH $t $LIMIT > $OUTPATH/ap-$t.txt &
    done
done

wait

tail out/clevr-state/$MODEL-*/ap-*.txt
