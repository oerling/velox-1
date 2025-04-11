
struct Depends {
  int32_t block;
  bool fuseIfsingle{false};
  bool fuseIfMultiple{false};
  needsResultOnHost{false};
};

struct BlockProxy {
  float latencyMicros(int32_t numBlocks);

  /// Rows to process. This is the maximum parallelism.
  int32_t numRows;

  /// Single block throughput. Running all blocks that fit at a time gets 1/3 of this per block, so going from 1 block to 480 blocks is 130x more throughput.
  float maxRowsPerMicroPerBlock;

  /// True if running this on many blocks needs a follow up kernel launch to add up results. E.g. a prefix sum on 10 blocks needs a kernel to add up the component sums and a third kernel to update the values from non-first blocks.
  bool multiBlockNeedsGather{false};
  
  std::vector<Depends> depends;
  
  // Describes the operation, e.g. GpuDecode or KernelBox
  void* block;
};




struct WaveProxy {

  int32_t numblocks;
  std::vector<BlockProxy> blocks;
};

/// Sequential operations in one kernel launch. This means running 'numBlocks' blocks where each block does ops[0], ops[1], etc. The ops may depend on the previous op and may have __syncthreads() barriers.
struct SequencePlan {
  /// Number of thread blocks.
  int32_t numBlocks;
  std::vector<BlockProxy*> ops;
};

/// describes a kernel launch where different blocks do different SequencePrograms. Every block is independent of every other block.
struct WavePlan {
  std::vector<SequencePlan> sequences;
};

/// Describes a sequence of consecutive kernel launches on one stream.
struct GridPlan {
  /// Guess of wall time latency if this is the only operation on device.
  float latencyMicros;
  std::vector<WavePlan> waves;
};


