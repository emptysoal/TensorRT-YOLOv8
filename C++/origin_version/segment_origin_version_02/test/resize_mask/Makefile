CUDA_PATH       = /usr/local/cuda
INCLUDE         = -I$(CUDA_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart

CC = nvcc

all: resize

resize: resize.cpp
	$(CC) -std=c++11 resize.cpp postprocess.cu -o resize $(INCLUDE) $(LDFLAG)

clean:
	rm -rf ./resize
