#!/bin/bash

MODEL=$1
LIMIT=$2  # how many files to process. specify smaller number to get faster results
USE_BASELINE=$3
THRESHOLDS="0.5 0.9 0.95 0.98 0.99"
ITERS="10 20 30"

if [ ! -z "$USE_BASELINE" ]
then
    ARGS="--decoder DSPN --encoder RNSumEncoder"
else
    ARGS="--decoder DSPN --encoder RNMaxEncoder"
fi

for iter in $ITERS; do
    OUTPATH="out/clevr-box/$MODEL-$iter"
    python train.py --show --loss hungarian --dim 512 --dataset clevr-box --epochs 100 --latent 512 --supervised --inner-lr 800 --resume logs/$MODEL --iters $iter --name test --eval-only --export-dir $OUTPATH --full-eval $ARGS --mask-feature

    for t in $THRESHOLDS; do
        scripts/eval-box.sh $OUTPATH $t $LIMIT > $OUTPATH/ap-$t.txt &
    done
done

wait

tail out/clevr-box/$MODEL-*/ap-*.txt
