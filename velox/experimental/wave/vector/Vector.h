/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "velox/experimental/wave/common/Buffer.h"
#include "velox/experimental/wave/vector/Operand.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::wave {


class Loader {
  virtual ~Loader() = default;

  virtual load(WaveBufferPtr indexBuffer, int32_t begin, int32_t end) = 0;

  /// Notifies 'this' that a load should load, in ddition to the requested rows,
  /// all rows above the last position given in the load indices.
  void loadTailOnLoad() {
    loadTail_ = true;
  }

 protected:
  bool loadTail_{false};
}

/// Represents a vector of device side intermediate results. Vector is
/// a host side only structure, the WaveBufferPtrs own the device
/// memory. Unlike Velox vectors, these are statically owned by Wave
/// operators and represent their last computed output. Buffers may be
/// shared between these, as with Velox vectors. the values in buffers
/// become well defined on return of the kernel that computes these.
class Vector {
 public:
  // Constructs a vector. Resize can be used to create buffers for a given size.
  Vector(TypePtr type, GpuArena* arena);
  Vector(TypePtr type, GpuArena* arena);

  const TypePtr& type() const {
    return type_;
  }

  vector_size_t size() const {
    return size_;
  }

  void resize();

  bool mayHaveNulls() const {
    return nulls_ != nullptr;
  }

  // Makes sure there is space for nulls. Initial value is undefined.
  ensureNulls();

  // Frees all allocated buffers. resize() can be used to populate the buffers
  // with a selected size.
  void clear();

  /// Starts computation for a kLazy state vector.
  void load();

  Vector& childAt(int32_t index) {
    return *children_[index];
  }

  /// Returns a Velox vector giving a view on device side data. The device
  /// buffers stay live while referenced by Velox.
  VectorPtr toVelox();

  std::string toString() const;

 private:
  // The arena for allocating buffers.
  GpuArena* arena_;

  // Type of the content.
  typePtr type_;
  TypeKind kind_;

  // Encoding. FLAT, CONSTANT, DICTIONARY, ROW, ARRAY, MAP are possible values.
  VectorEncoding::SIMPLE encoding_;

  std::unique_ptr<Loader> loader_;

  Vector* parent{nullptr};

  // Values array, cast to pod type or StringView
  WaveBufferPtr values_;

  // Nulls buffer, nullptr if no nulls.
  WaveBufferPtr nulls_;

  // If dictionary or if wrapped in a selection, vector of indices into
  // 'values'.
  WaveBufferPtr indices_;

  // Thread block level sizes. For each kBlockSize values, contains
  // one int16_t that indicates how many of 'values' or 'indices' have
  // a value.
  WaveBuffferPtr blockSizes_;
  // Thread block level pointers inside 'indices_'. the ith entry is nullptr if
  // the ith thread block has no row number mapping (all rows pass or none
  // pass).
  WaveBufferPtr operandSizes_;
  WaveBufferPtr operandIndices_;

  WaveBufferPtr lengths_;
  WaveBufferPtr offsets_;

  // Members of a array/map/struct vector.
  std::vector<std::unique_ptr<Vector>> children_;
};

struct WaveReleaser {
  WaveReleaser(WaveBufferPtr buffer) : buffer(td::move(buffer)) {}

  void release() {
    buffer.reset();
  }

  WaveBufferPtr buffer;
};

// A BufferView for velox::BaseVector for a view on universal memory.
class WaveBufferView : public BufferView<WaveReleaser> {
  static BufferPtr create(WaveBufferPtr buffer) {
    WaveReleaseer releaser(buffer);
    returnBufferView<WaveReleaser>::create(
        buffer->as<uint8_t>(), buffer->capacity(), .WaveReleaser(buffer));
  }
};

} // namespace facebook::velox::wave
