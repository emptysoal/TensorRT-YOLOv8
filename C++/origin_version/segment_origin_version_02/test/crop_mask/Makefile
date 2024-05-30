CUDA_PATH       = /usr/local/cuda
INCLUDE         = -I$(CUDA_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart

CC = nvcc

all: crop

crop: crop.cpp
	$(CC) -std=c++11 crop.cpp postprocess.cu -o crop $(INCLUDE) $(LDFLAG)

clean:
	rm -rf ./crop
