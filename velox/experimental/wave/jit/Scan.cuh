
#pragma once

#include "velox/experimental/wave/jit/BitUtil.cuh"

namespace facebook::velox::wave {

template <typename T, int32_t kNumLanes = kWarpThreads>
struct WarpScan {
  enum {
    /// Whether the logical warp size and the PTX warp size coincide
    IS_ARCH_WARP = (kNumLanes == kWarpThreads),

    /// The number of warp scan steps
    STEPS = Log2<kNumLanes>::VALUE,

    /// The 5-bit SHFL mask for logically splitting warps into sub-segments
    /// starts 8-bits up
    SHFL_C = (kWarpThreads - kNumLanes) << 8

  };

  static constexpr unsigned int member_mask =
      kNumLanes == 32 ? 0xffffffff : (1 << kNumLanes) - 1;

  WarpScan() = default;

  __device__ __forceinline__ void exclusiveSum(
      T input, ///< [in] Calling thread's input item.
      T& exclusive_output) ///< [out] Calling thread's output item.  May be
                           ///< aliased with \p input.
  {
    T initial_value = 0;
    exclusiveSum(input, exclusive_output, initial_value);
  }

  __device__ __forceinline__ void exclusiveSum(
      T input, ///< [in] Calling thread's input item.
      T& exclusive_output, ///< [out] Calling thread's output item.  May be
                           ///< aliased with \p input.
      T initial_value) {
    T inclusive_output;
    inclusiveSum(input, inclusive_output);

    exclusive_output = initial_value + inclusive_output - input;
  }

  __device__ __forceinline__ void ExclusiveSum(
      T input, 
      T& exclusive_output, 
      T initial_value,
      T& warp_aggregate) 
  {
    T inclusive_output;
    Inclusivesum(input, inclusive_output);
    warp_aggregate = __shfl_sync(member_mask, inclusive_output, kNumLanes - 1);
    exclusive_output = initial_value + inclusive_output - input;
  }

  __device__ __forceinline__ unsigned int inclusiveSumStep(
      unsigned int input,
      int first_lane, ///< [in] Index of first lane in segment
      int offset) ///< [in] Up-offset to pull from
  {
    unsigned int output;
    int shfl_c = first_lane | SHFL_C; // Shuffle control (mask and first-lane)

    // Use predicate set from SHFL to guard against invalid peers
    asm volatile(
        "{"
        "  .reg .u32 r0;"
        "  .reg .pred p;"
        "  shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
        "  @p add.u32 r0, r0, %4;"
        "  mov.u32 %0, r0;"
        "}"
        : "=r"(output)
        : "r"(input), "r"(offset), "r"(shfl_c), "r"(input), "r"(member_mask));

    return output;
  }

  __device__ __forceinline__ void inclusiveSum(
      T input, 
      T& inclusive_output) {
    inclusive_output = input;
    int segment_first_lane = 0;
#pragma unroll
    for (int STEP = 0; STEP < STEPS; STEP++) {
      inclusive_output = inclusiveSumStep(
          inclusive_output, segment_first_lane, (1 << STEP));
    }
  }
};

} // namespace facebook::velox::wave
