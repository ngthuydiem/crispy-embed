#!/bin/zsh

cd ..
INPUT="crispy-embed/data/V2Mice_sample1000"
K=6
CUTOFF=0.3

./crispy-embed/run-crispy.sh $INPUT $K $CUTOFF

