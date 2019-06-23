#!/bin/bash

python Object-Detection-Metrics-State/pascalvoc.py -gt ../$1/groundtruths -det ../$1/detections -t $2 -np --limit $3
