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
#include "velox/vector/BaseVector.h"

namespace facebook::velox::wave {

/// Represents a vector of device side intermediate results. Vector is
/// a host side only structure, the WaveBufferPtrs own the device
/// memory. Unlike Velox vectors, these are statically owned by Wave
/// operators and represent their last computed output. Buffers may be
/// shared between these, as with Velox vectors. the values in buffers
/// become well defined on return of the kernel that computes these. A
/// Wave operator's output is a TypeKind::ROW Vector. Different waves
/// may produce values for different children at different times. Each
/// child has a separate callback for arrival of data. This is called
/// on the host to schedule data dependent activity. An expression
/// depending on a column becomes executable when the column has a
/// value. An operation that depends on all the columns becomes
/// executable when all children of a complex type have a value.
class Vector {
 public:
  // Constructs a vector. Resize can be used to create buffers for a given size.
  Vector(TypePtr type, GpuArena* arena) : type_(type), arena_(arena) {}

  const TypePtr& type() const {
    return type_;
  }

  vector_size_t size() const {
    return size_;
  }

  void resize();

  bool mayHaveNulls() const {
    return nulls_ != nullptr || dictionaryNulls_ != nullptr;
  }

  // Makes sure there is space for nulls. Initial value is undefined.
  ensureNulls();

  // Frees all allocated buffers. resize() can be used to populate the buffers
  // with a selected size.
  void reset();

  Vector& childAt(int32_t index) {
    return *children_[index];
  }

  /// Calls 'callback' when 'subfields' all have a value. If 'subfields' is
  /// empty, 'callback' is called when all fields of 'this' have a value.
  void onArrival(
      const std::vector<Subfield>& subfields,
      std::function<void()> claaback); );

  // Informs the producer that the consumer no longer references'ths' or
  // children.
  void consumed();

  /// Returns or creates a child of the given name and type. This can
  /// happen with flat map as struct. A getter depends on a key which
  /// may or may not be present in the struct representing a flat
  /// map. 'name' is cast to the appropriate number if 'this' is a
  /// flat map struct with a numeric key.
  Vector& namedChild(const std::string& name);

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

  // If 'typeKind_' is MAP, and 'flatMap_' is true, 'this' is a struct style
  // flat map where the key maps to an index in 'children_'.
  bool flatMap_{false};

  // map from key to index in 'children_' for  string flat map keys.
  std::shared_ptr<folly::F14FastMap<std::string, int32_t>> stringKeys_;

  // map from key to index in 'children_' for flat maps with numeric keys.
  std::shared_ptr<folly::F14FastMap<uint64_t, int32_t>> numericKeys_;

  // Count of children presently being computed. For a leaf vector, this goes
  // from 1 to 0 when the value arrives.
  std::atomic<int32_t> numPendingChildren_{0};

  ArrivedCallback arrived_;

  // Count of children that have a value but have not been fully consumed by
  // downstream. For complex type, when this goes to zero the complex type has
  // been consumed. For leaf types the leaf is consumed when this is 1.
  std::atomic<int32_t> consumedChildren_{0};
  ConsumedCallback fullyConsumed_;

  // A vector in a complex type has exactly one parent. The child becoming ready
  // decrements the pending children count of the parent, which in turn becomes
  // ready when the pending children count goes to zero.
  Vector* parent{nullptr};

  // Values array, cast to pod type or StringView
  WaveBufferPtr values_;

  // Nulls buffer, nullptr if no nulls.
  WaveBufferPtr nulls_;

  // In dictionary encoding, a single level of indices over 'values'. There are
  // no dictionaries over dictionaries.
  WaveBufferPtr indices_;

  // If non-null, bit mask with for nulls added by the dictionary.
  WaveBufferPtr dictionaryNulls_;

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
