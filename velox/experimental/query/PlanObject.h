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

#include "velox/experimental/query/QueryGraphContext.h"

namespace facebook::verax {

/// Enum for types of query graph nodes.
enum class PlanType {
  kTable,
  kDerivedTable,
  kColumn,
  kLiteral,
  kCall,
  kAggregate,
  kProject,
  kFilter
};

Name planTypeName(PlanType type);

inline bool isExprType(PlanType type) {
  return type == PlanType::kColumn || type == PlanType::kCall ||
      type == PlanType::kLiteral;
}


  
struct PlanObject {
  PlanObject(PlanType _type) : type(_type) {
    id = queryCtx()->newId(this);
  }

  void operator delete(void* ptr) {
    LOG(FATAL) << "Plan objects are not deletable";
  }

  template <typename T>
  T as() {
    return reinterpret_cast<T>(this);
  }

  template <typename T>
  const T as() const {
    return reinterpret_cast<const T>(this);
  }

  virtual PtrSpan<PlanObject> children() const {
    return PtrSpan<PlanObject>(nullptr, nullptr);
  }

  template <typename Func>
  void preorderVisit(Func func) {
    func(this);
    for (auto child : children()) {
      child->preorderVisit(func);
    }
  }

  virtual bool isExpr() const {
    return false;
  }

  size_t hash() const;

  virtual std::string toString() const {
    return fmt::format("#{}", id);
  }
  PlanType type;
  int32_t id;
};

  class PlanObjectSet {
 public:
  bool contains(PlanObjectConstPtr object) const {
    return object->id < bits_.size() * 64 &&
        velox::bits::isBitSet(bits_.data(), object->id);
  }

  bool operator==(const PlanObjectSet& other) const;

  size_t hash() const;

  void add(PlanObjectPtr ptr) {
    auto id = ptr->id;
    ensureSize(id);
    velox::bits::setBit(bits_.data(), id);
  }

  /// Returns true if 'this' is a subset of 'super'.
  bool isSubset(const PlanObjectSet& super) const;

  void erase(PlanObjectPtr object) {
    if (object->id < bits_.size() * 64) {
      velox::bits::clearBit(bits_.data(), object->id);
    }
  }

  void unionColumns(ExprPtr expr);

  void unionColumns(const ExprVector& exprs);

  void unionSet(const PlanObjectSet& other);

  void intersect(const PlanObjectSet& other);

  template <typename V>
  void unionObjects(const V& objects) {
    for (PlanObjectPtr& object : objects) {
      add(object);
    }
  }

  template <typename Func>
  void forEach(Func func) const {
    auto ctx = queryCtx();
    velox::bits::forEachSetBit(bits_.data(), 0, bits_.size() * 64, [&](auto i) {
      func(ctx->objectAt(i));
    });
  }

  template <typename T = PlanObjectPtr>
  std::vector<T> objects() const {
    std::vector<T> result;
    forEach(
        [&](auto object) { result.push_back(reinterpret_cast<T>(object)); });
    return result;
  }

  std::string toString(bool names) const;

 private:
  void ensureSize(int32_t id) {
    ensureWords(velox::bits::nwords(id + 1));
  }

  void ensureWords(int32_t size) {
    if (bits_.size() < size) {
      bits_.resize(size);
    }
  }

  std::vector<uint64_t, velox::StlAllocator<uint64_t>> bits_{stl<uint64_t>()};
};

}
