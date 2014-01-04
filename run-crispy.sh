#!/bin/zsh

#input: $INPUT, $K_CUTOFF, $CUTOFF, $B
PROFILER=(/usr/bin/time -f "%e %M")
INPUT=$1
K=$2

UNIQUE_INPUT=$INPUT"_Clean"

./crispy-embed/bin/preprocess -i $INPUT
./crispy-embed/bin/findSeeds -i $UNIQUE_INPUT -k $K -s 1
./crispy-embed/bin/kmerDist -i $UNIQUE_INPUT -k $K -c

NUM_READS=`grep '>' $UNIQUE_INPUT | wc -l`
$PROFILER ./crispy-embed/bin/euclidDist -i $UNIQUE_INPUT -n $NUM_READS
NUM_FILES=`ls $UNIQUE_INPUT".edist"* | wc -l`	
$PROFILER ./crispy-embed/bin/aveclust -i $UNIQUE_INPUT -n $NUM_READS -f $NUM_FILES 

