CUDA_DIR = /usr/local/cuda

ARCH = sm_20	#use sm_20 for Fermi architecture
NVCC = nvcc
CXX = $(NVCC)

NVCCOPTIONS = -arch $(ARCH) --ptxas-options=-v --compiler-options=-Wall --compiler-options=-Wno-unused-result
NVCCFLAGS = -O3 $(NVCCOPTIONS) -Xcompiler -fopenmp -I$(CUDA_DIR)/include -L$(MPI_DIR)/lib -L$(CUDA_DIR)/lib64

CXXFLAGS = $(NVCCFLAGS)
NVCCLIBS = -lm -lgomp -lcudart 

CXXOBJS = kmerMain.o kmerCPU.o 
NVCCOBJS =  kmerKernel.o kmerCUDA.o
EXEC = kmerDist

all: $(CXXOBJS) $(NVCCOBJS)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $(CXXOBJS) $(NVCCOBJS) $(NVCCLIBS)
	strip $(EXEC)
kmerKernel.o:
	$(NVCC) $(NVCCFLAGS) -c kmerKernel.cu 
kmerCUDA.o: 
	$(NVCC) $(NVCCFLAGS) -c kmerCUDA.cu 
clean: 
	rm -rf $(EXEC)  *.o 


