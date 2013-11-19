#!/bin/zsh

INPUT="data/V2Mice_sample1000"
UNIQUE_INPUT=$INPUT"_Clean"

THRESHOLD=0.07
K=5
START=1
END=8

echo "#########################  CRISPY-EMBED (GPU) #########################"

for (( K=$START; K<=$END; K++ )) 
do
echo $K
START_TIME=$(date +%s.%N)
./bin/preprocess -i $INPUT 
./bin/findSeeds -i $UNIQUE_INPUT -k $K -s 2 -o $UNIQUE_INPUT".pair" 
./bin/kmerDist -i $UNIQUE_INPUT -k $K

NUM_READS=`grep '>' $UNIQUE_INPUT | wc -l`
echo $NUM_READS
NUM_ENTRIES=`./bin/euclidDist -i $UNIQUE_INPUT -r $NUM_READS -t $THRESHOLD`
NUM_FILES=`ls $UNIQUE_INPUT".edist"* | wc -l`	

./bin/aveclust -i $UNIQUE_INPUT -m $NUM_ENTRIES -n $NUM_READS -f $NUM_FILES -s 0.01 -e $THRESHOLD

END_TIME=$(date +%s.%N)
DIFF=$(echo " $END_TIME - $START_TIME " | bc)

(( SPARSITY = $NUM_ENTRIES * 1.0 / ($NUM_READS * ($NUM_READS - 1) / 2) ))
echo "Sparse matrix contains "$NUM_ENTRIES" entries in "$NUM_FILES" files with sparsity "$SPARSITY
echo "It took $DIFF seconds"

python2.7 ../parse-esprit.py -i $UNIQUE_INPUT".Cluster"
python2.7 ../compute-accuracy.py $UNIQUE_INPUT  $UNIQUE_INPUT".Cluster.Formatted"

rm -f $UNIQUE_INPUT $UNIQUE_INPUT"."*
done


