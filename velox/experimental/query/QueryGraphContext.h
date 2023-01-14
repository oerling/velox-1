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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/core/PlanNode.h"

namespace facebook::verax {

/// Base data structures for plan candidate generation.

using Name = const char*;

template <typename T>
using PtrSpan = folly::Range<T**>;

struct PlanObject;

using PlanObjectPtr = PlanObject* FOLLY_NONNULL;
using PlanObjectConstPtr = const PlanObject* FOLLY_NONNULL;

struct PlanObjectPtrHasher {
  size_t operator()(const PlanObjectPtr& object) const;
};

struct PlanObjectPtrComparer {
  bool operator()(const PlanObjectPtr& lhs, const PlanObjectPtr& rhs) const;
};

class Plan;

class QueryGraphContext {
 public:
  QueryGraphContext(velox::HashStringAllocator& allocator)
      : allocator_(allocator),
        stlAllocator_(velox::StlAllocator<void*>(&allocator)) {}

  Name toName(std::string_view str);

  int32_t newId(PlanObject* FOLLY_NONNULL object) {
    objects_.push_back(object);
    return objects_.size() - 1;
  }

  velox::StlAllocator<void*>* stlAllocator() & {
    return &stlAllocator_;
  }

  velox::HashStringAllocator& allocator() {
    return allocator_;
  }

  velox::HashStringAllocator& allocator_;
  velox::StlAllocator<void*> stlAllocator_;

  /// Returns a canonical instance for all logically equal values of 'object'.
  /// Returns 'object' on first call with object, thereafter the same physical
  /// object if the argument is equal.
  PlanObjectPtr dedup(PlanObjectPtr object);

  PlanObjectPtr objectAt(int32_t id) {
    return objects_[id];
  }

  /// Returns the top level plan being processed when printing operator trees.
  /// If non-null, allows showing percentages.
  Plan*& contextPlan() {
    return contextPlan_;
  }

  // PlanObjects are stored at the index given by their id.
  std::vector<PlanObjectPtr> objects_;
  std::unordered_set<std::string_view> names_;
  std::unordered_set<PlanObjectPtr, PlanObjectPtrHasher, PlanObjectPtrComparer>
      deduppedObjects_;
  Plan* FOLLY_NULLABLE contextPlan_{nullptr};
};

inline QueryGraphContext*& queryCtx() {
  thread_local QueryGraphContext* context;
  return context;
}

template <typename T>
velox::StlAllocator<T> stl() {
  return *reinterpret_cast<velox::StlAllocator<T>*>(queryCtx()->stlAllocator());
}

#define Declare(T, destination, ...)                          \
  T* destination = reinterpret_cast<T*>(                     \
      queryCtx()->allocator().allocate(sizeof(T))->begin()); \
  new (destination) T(__VA_ARGS__);

#define DeclaretDefault(T, destination)                        \
  T* destination = reinterpret_cast<T*>(                     \
      queryCtx()->allocator().allocate(sizeof(T))->begin()); \
  new (destination) T();

/// Converts std::string to name used in query graph objects. raw pointer to
/// arena allocated const chars.
Name toName(const std::string& string);


struct Expr;
using ExprPtr = Expr*;
struct Column;
using ColumnPtr = Column*;
using ExprVector = std::vector<ExprPtr, velox::StlAllocator<ExprPtr>>;
using ColumnVector = std::vector<ColumnPtr, velox::StlAllocator<ColumnPtr>>;
 
} // namespace facebook::verax
