CUDA_DIR = /usr/local/cuda

ARCH = sm_20	#use sm_20 for Fermi architecture
NVCC = nvcc
CXX = $(NVCC)

NVCCOPTIONS = --generate-code arch=compute_20,code=sm_20 --ptxas-options=-v --compiler-options=-Wall --compiler-options=-Wno-unused-result
NVCCFLAGS = -O3 $(NVCCOPTIONS) -Xcompiler -fopenmp -I$(CUDA_DIR)/include -L$(MPI_DIR)/lib -L$(CUDA_DIR)/lib64 -lineinfo -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA

CXXFLAGS = $(NVCCFLAGS)
NVCCLIBS = -lm -lgomp -lcudart 

CXXOBJS = euclidMain.o euclidCPU.o 
NVCCOBJS =  euclidKernel.o euclidCUDA.o 
EXEC = euclidDist

all: $(CXXOBJS) $(NVCCOBJS)
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) $(CXXOBJS) $(NVCCOBJS) $(NVCCLIBS)
	strip $(EXEC)
euclidKernel.o:
	$(NVCC) $(NVCCFLAGS) -c euclidKernel.cu 
euclidCUDA.o: 
	$(NVCC) $(NVCCFLAGS) -c euclidCUDA.cu 
clean: 
	rm -rf $(EXEC)  *.o 


