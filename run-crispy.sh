#!/bin/zsh

#INPUT="data/V2Mice_sample30000"
#UNIQUE_INPUT=$INPUT"_Clean"

STEP_SIZE=10
THRESHOLD=1.0
K=2

START=1
END=1

for (( i=$START; i<=$END; i++ )) 
do
(( NUM_READS = $i * $STEP_SIZE ))
NAME="microbial_"$NUM_READS

INPUT="/DATA/grinder_data"/$NAME"-reads.fa"
OUTPUT="/DATA/grinder_mat"/$NAME
UNIQUE_INPUT=$INPUT

echo "#########################  CRISPY-EMBED (GPU) #########################"

START_TIME=$(date +%s.%N)
#./bin/preprocess -i $INPUT 
./bin/findSeeds -i $UNIQUE_INPUT -o $UNIQUE_INPUT".pair" -k $K -s 2 -t 0.03
./bin/kmerDist -i $UNIQUE_INPUT -k $K -c

NUM_READS=`grep '>' $UNIQUE_INPUT | wc -l`
NUM_ENTRIES=`./bin/euclidDist -i $UNIQUE_INPUT -r $NUM_READS -t $THRESHOLD`
NUM_FILES=`ls $UNIQUE_INPUT".edist"* | wc -l`	

./bin/aveclust -i $UNIQUE_INPUT -m $NUM_ENTRIES -n $NUM_READS -f $NUM_FILES -o $OUTPUT -s 0.1 -e $THRESHOLD

END_TIME=$(date +%s.%N)
DIFF=$(echo " $END_TIME - $START_TIME " | bc)

(( SPARSITY = $NUM_ENTRIES * 1.0 / ($NUM_READS * ($NUM_READS - 1) / 2) ))
echo "Sparse matrix contains "$NUM_ENTRIES" entries in "$NUM_FILES" files with sparsity "$SPARSITY
echo "It took $DIFF seconds"

#rm -f $INPUT"."*

done


