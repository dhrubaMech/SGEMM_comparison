
NVCC      = nvcc
NVCCFLAGS = -O3 -Xcompiler -fopenmp  -Wno-deprecated-gpu-targets -lineinfo 
LDFLAGS   = -lcudart

CXX      = g++
CXXFLAGS = -fopenmp

TARGET  = res
SOURCES = matMul.cu helperFunctions.cpp
OBJECTS = matMul.o helperFunctions.o

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) 


matMul.o: matMul.cu
	$(NVCC) $(NVCCFLAGS) -c matMul.cu -o matMul.o

helperFunctions.o: helperFunctions.cpp
	$(NVCC) $(NVCCFLAGS) -c helperFunctions.cpp -o helperFunctions.o

clean:
	rm -f $(OBJECTS) $(TARGET)

