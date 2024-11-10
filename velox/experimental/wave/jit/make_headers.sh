#/bin/sh
# Generates the inlined headers for Wave Jit.
#
# Run in the valox checkout root. 

JIT=velox/experimental/wave/jit

head --lines 16 velox/experimental/wave/common/Cuda.h > $JIT/Headers.cpp
echo "namespace facebook::velox::wave {" >> $JIT/Headers.cpp

echo "void registerHeader(const char* text);" >> $JIT/Headers.cpp

stringify $JIT/BitUtils.cuh >> $JIT/Headers.cpp
stringify $JIT/Scan.cuh >> $JIT/Headers.cpp
stringify "velox/experimental/wave/exec/WaveCore.cuh" >> $JIT/Headers.cpp
stringify "velox/experimental/wave/exec/ExprKernel.h" >> $JIT/Headers.cpp
stringify "velox/experimental/wave/common/hashTable.h" >> $JIT/Headers.cpp
stringify "velox/experimental/wave/common/hashTable.cuh" >> $JIT/Headers.cpp
stringify "velox/experimental/wave/common/hash.cuh" >> $JIT/Headers.cpp
stringify "velox/experimental/wave/common/StringView.cuh" >> $JIT/Headers.cpp

echo "}" >> $JIT/Headers.cpp

clang-format -i $JIT/Headers.cpp


