#/bin/sh
# Generates the inlined headers for Wave Jit.
#
# Run in the valox checkout root. 

JIT=velox/experimental/wave/jit

head --lines 16 velox/experimental/wave/common/Cuda.h > $JIT/Headers.h
echo "namespace facebook::velox::wave {" >> $JIT/Headers.h

echo "bool registerHeader(const char* text);" >> $JIT/Headers.h

stringify $JIT/BitUtil.cuh >> $JIT/Headers.h
stringify $JIT/Scan.cuh >> $JIT/Headers.h
stringify "velox/experimental/wave/exec/WaveCore.cuh" >> $JIT/Headers.h
stringify "velox/experimental/wave/exec/ExprKernel.h" >> $JIT/Headers.h
stringify "velox/experimental/wave/common/HashTable.h" >> $JIT/Headers.h
stringify "velox/experimental/wave/common/HashTable.cuh" >> $JIT/Headers.h
stringify "velox/experimental/wave/common/hash.cuh" >> $JIT/Headers.h
stringify "velox/experimental/wave/common/StringView.cuh" >> $JIT/Headers.h
stringify "velox/experimental/wave/common/StringView.h" >> $JIT/Headers.h
stringify "velox/experimental/wave/common/Hash.h" >> $JIT/Headers.h
stringify "velox/experimental/wave/common/CompilerDefines.h" >> $JIT/Headers.h

echo "}" >> $JIT/Headers.h

clang-format -i $JIT/Headers.h


