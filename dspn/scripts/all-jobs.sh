#!/bin/bash

set -x

DATASETS="clevr-box clevr-state"
NUMS="1"  # results in paper produced with NUMS="1 2 3 4 5 6"

for num in $NUMS; do
    for dataset in $DATASETS; do
        scripts/clevr.sh $dataset $num
        scripts/clevr.sh $dataset $num baseline
    done
done
