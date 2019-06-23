#!/bin/bash

python Object-Detection-Metrics/pascalvoc.py -gt ../$1/groundtruths -det ../$1/detections -t $2 -np -gtformat xyrb -detformat xyrb --limit $3
