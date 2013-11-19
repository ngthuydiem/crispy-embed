rm -r bin
mkdir bin

cd src/preprocess
make clean
make
cp preprocess ../../bin
make clean
cd ../..

cd src/findSeeds
make clean
make
cp findSeeds ../../bin
make clean
cd ../..

cd src/kmerDist
make clean
make
cp kmerDist ../../bin
make clean
cd ../..

cd src/euclidDist
make clean
make
cp euclidDist ../../bin
make clean
cd ../..

cd src/aveclust
make clean
make
cp aveclust ../../bin
make clean
cd ../..
