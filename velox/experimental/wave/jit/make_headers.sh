# Generates the inlined headers for Wave Jit.
#
# Run in the valox checkout root. 

PATH=velox/experimental/wave/jit

echo "namespace facebook::velox::wave {" > $PATH/Headers.cpp

echo "void registerHeader(const char* text);" >> $PATH/Headers.cpp

stringify $PATH/BitUtils.cuh >> $PATH/Headers.cpp
stringify $PATH/Scan.cuh >> $PATH/Headers.cpp
stringify "velox/experimental/wave/exec/WaveCore.cuh" >> $PATH/Headers.cpp
stringify "velox/experimental/wave/exec/ExprKernel.h" >> $PATH/Headers.cpp
stringify "velox/experimental/wave/common/hashTable.h" >> $PATH/Headers.cpp
stringify "velox/experimental/wave/common/hashTable.cuh" >> $PATH/Headers.cpp
stringify "velox/experimental/wave/common/hash.cuh" >> $PATH/Headers.cpp
stringify "velox/experimental/wave/common/StringView.cuh" >> $PATH/Headers.cpp


echo "}" >> $PATH/Headers.cpp



