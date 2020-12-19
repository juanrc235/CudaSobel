CXX=nvcc

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv4`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib -lcudart `pkg-config --libs opencv4`

all:
	$(CXX) $(CFLAGS) $(LDFLAGS) -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets main.cu -o CONVOLUTIONer 

clean:
	rm -f CONVOLUTIONer 

