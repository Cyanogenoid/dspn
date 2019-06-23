#!/bin/bash

DATASET=$1
NUM=$2
USE_BASELINE=$3

if [ ! -z "$USE_BASELINE" ]
then
        PREFIX="base"
else
        PREFIX="dspn"
fi

ARGS="$PREFIX-$DATASET-$NUM 100000 $USE_BASELINE"
if [[ $DATASET == *"box"* ]]; then
	scripts/eval-box-model.sh $ARGS
else
	scripts/eval-state-model.sh $ARGS
fi
