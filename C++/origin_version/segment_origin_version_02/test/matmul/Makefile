CUDA_PATH       = /usr/local/cuda
INCLUDE         = -I$(CUDA_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart

CC = nvcc

all: matmul

matmul: matmul.cpp
	$(CC) -std=c++11 matmul.cpp postprocess.cu -o matmul $(INCLUDE) $(LDFLAG)

clean:
	rm -rf ./matmul
