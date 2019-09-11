#!/bin/bash

DATASETS="clevr-box clevr-state"
NUMS="4 5 6"

for num in $NUMS; do
	for dataset in $DATASETS; do
        scripts/eval-clevr.sh $dataset $num
        scripts/eval-clevr.sh $dataset $num baseline
	done
done
