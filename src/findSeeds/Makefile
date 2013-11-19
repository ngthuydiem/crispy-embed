CXX = g++ -O3 -Wall -Wextra -Wno-unused-result -std=gnu++0x -fopenmp

EXEC = findSeeds

CXXOBJS = findSeeds.o

all: $(CXXOBJS)
	$(CXX) -o $(EXEC) $(CXXOBJS)
	strip $(EXEC)
clean: 
	rm -rf $(EXEC)  *.o 


