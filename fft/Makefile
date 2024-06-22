CUDA_PATH 	  ?= /usr/local/cuda-12.4
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc
INCLUDE ?= $(CUDA_PATH)/include 
LIB ?= $(CUDA_PATH)/lib -lcufft

all: build

build: myFft cuFft

myFft:	myFft.cu 
	$(NVCC) myFft.cu -o myFft -I$(INCLUDE) -L$(LIB)
cuFft: 	cuFft.cu
	$(NVCC) cuFft.cu -o cuFft -I$(INCLUDE) -L$(LIB)
clean:
	rm -f *Fft 