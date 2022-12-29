
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

#include "velox/experimental/query/Costs.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/experimental/query/PlanToGraph.h"
#include "velox/experimental/query/PlanUtils.h"

namespace facebook::verax {

using namespace facebook::velox;

// Collection of per operation costs for a target system.  The base
// unit is the time to memcpy a cache line in a large memcpy on one
// core. This is ~6GB/s, so ~10ns. Other times are expressed as
// multiples of that.
struct Costs {
  static float byteShuffleCost() {
    return 12; // ~500MB/s
  }

  static float hashProbeCost(float cardinality) {
    return cardinality < 10000 ? kArrayProbeCost
        : cardinality < 500000 ? kSmallHashCost
                               : kLargeHashCost;
  }

  static constexpr float kArrayProbeCost = 2; // ~10 instructions.
  static constexpr float kSmallHashCost = 10; // 50 instructions
  static constexpr float kLargeHashCost = 40; // 2 LLC misses
};

void GroupBy::setCost() {
  float cardinality = 1;
  for (auto key : grouping) {
    cardinality *= key->value.cardinality;
  }
  auto inputCardinality = input->inputCardinality * input->fanout;
  // The estimated output is input minus the times an input is a duplicate of a
  // key already in the input. The probability of a duplicate is approximated as
  // (1 - (1 / d))^n. where d is the number of potentially distinct keys  and n
  // is the number of keys in the input. This approaches d as n goes to
  // infinity.
  auto numDuplicate =
      inputCardinality * pow(1.0 - (1.0 / cardinality), inputCardinality);
  auto nOut = inputCardinality - numDuplicate;
  fanout = nOut / inputCardinality;
  unitCost =
      grouping.size() * Costs::hashProbeCost(inputCardinality - numDuplicate);
}

template <typename V>
float shuffleCostV(const V& columns) {
  int32_t size = 0;
  for (auto column : columns) {
    size += column->value.byteSize();
  }
  return size * Costs::byteShuffleCost();
}

float shuffleCost(const ColumnVector& columns) {
  return shuffleCostV(columns);
}

float shuffleCost(const ExprVector& columns) {
return shuffleCostV(columns);
}

void Repartition::setCost() {
  unitCost = shuffleCost(columns);
}

} // namespace facebook::verax
