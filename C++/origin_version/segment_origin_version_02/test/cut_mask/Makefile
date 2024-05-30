CUDA_PATH       = /usr/local/cuda
INCLUDE         = -I$(CUDA_PATH)/include
LDFLAG          = -L$(CUDA_PATH)/lib64 -lcudart

CC = nvcc

all: cut

cut: cut.cpp
	$(CC) -std=c++11 cut.cpp postprocess.cu -o cut $(INCLUDE) $(LDFLAG)

clean:
	rm -rf ./cut
