CXXFLAGS := -std=c++17 -g 
#NVFLAGS= -g -G 
NVFLAGS= -g -O3 
TARGETS := bandwidthtest async transpose async2 indicesofsetbits

indicesofsetbits: IndicesOfSetBitsCuda.o IndicesOfSetBits.o
	nvcc $(NVFLAGS) $(OUTPUT_OPTION) $^

IndicesOfSetBitsCuda.o: IndicesOfSetBitsCuda.cu
	nvcc  $(NVFLAGS) $(OUTPUT_OPTION) $(CXXFLAGS) -lineinfo -c $<

IndicesOfSetBits.o: IndicesOfSetBits.cc IndicesOfSetBits.h
	gcc $(OUTPUT_OPTION) $(CXXFLAGS) -c -mavx2 $<

%: %.cu
	nvcc  $(NVFLAGS) $(OUTPUT_OPTION) $(CXXFLAGS) $<

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	$(RM) $(TARGETS) *.o
