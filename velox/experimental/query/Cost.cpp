
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

#include "velox/experimental/query/Cost.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/experimental/query/Plan.h"
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

  static constexpr float kKeyCompareCost =
      6; // ~30 instructions to find, decode and an compare
  static constexpr float kArrayProbeCost = 2; // ~10 instructions.
  static constexpr float kSmallHashCost = 10; // 50 instructions
  static constexpr float kLargeHashCost = 40; // 2 LLC misses
  static constexpr float kColumnRowCost = 5;
  static constexpr float kColumnByteCost = 0.1;

  // Cost of hash function on one column.
  static constexpr float kHashColumnCost = 0.5;

  // Cost of getting a column from a hash table
  static constexpr float kHashExtractColumnCost = 0.5;
};

void RelationOp::setCost(const PlanState& state) {
  inputCardinality = state.fanout;
}

float Index::lookupCost(float range) {
  return Costs::kKeyCompareCost * log(range) / log(2);
}

float orderPrefixDistance(
    RelationOpPtr input,
    IndexPtr index,
    const ExprVector& keys) {
  int32_t i = 0;
  float selection = 1;
  for (; i < input->distribution.order.size() &&
       i < index->distribution.order.size() && i < keys.size();
       ++i) {
    if (input->distribution.order[i]->sameOrEqual(*keys[i])) {
      selection *= index->distribution.order[i]->value.cardinality;
    }
  }
  return selection;
}

void TableScan::setCost(const PlanState& input) {
  RelationOp::setCost(input);
  float size = 0;
  for (auto& column : columns) {
    size += column->value.byteSize();
  }

  if (!keys.empty()) {
    float lookupRange(index->distribution.cardinality);
    float orderSelectivity = orderPrefixDistance(this->input, index, keys);
    auto distance = lookupRange * orderSelectivity;
    float batchSize = std::min<float>(inputCardinality, 10000);
    if (orderSelectivity == 1) {
      // The data does not come in key order.
      float batchCost = index->lookupCost(lookupRange) +
          index->lookupCost(lookupRange / batchSize) *
              std::max<float>(1, batchSize);
      unitCost = batchCost / batchSize;
    } else {
      float batchCost = index->lookupCost(lookupRange) +
          index->lookupCost(distance) * std::max<float>(1, batchSize);
      unitCost = batchCost / batchSize;
    }
    return;
  } else {
    fanout = index->distribution.cardinality * baseTable->filterSelectivity;
  }
  auto numColumns = columns.size();
  auto rowCost = numColumns * Costs::kColumnRowCost +
      std::max<float>(0, size - 8 * numColumns) * Costs::kColumnByteCost;
  unitCost += fanout * rowCost;
}

void Aggregation::setCost(const PlanState& input) {
  RelationOp::setCost(input);
  float cardinality = 1;
  for (auto key : grouping) {
    cardinality *= key->value.cardinality;
  }
  auto inputCardinality = this->input->inputCardinality * this->input->fanout;
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

void Repartition::setCost(const PlanState& input) {
  RelationOp::setCost(input);

  unitCost = shuffleCost(columns);
}

void HashBuild::setCost(const PlanState& input) {
  unitCost = keys.size() * Costs::kHashColumnCost +
      Costs::hashProbeCost(inputCardinality) +
      this->input->columns.size() * Costs::kHashExtractColumnCost * 2;
}

void JoinOp::setCost(const PlanState& input) {
  RelationOp::setCost(input);
  float buildSize = right->inputCardinality;
  auto rowCost = right->input->columns.size() * Costs::kHashExtractColumnCost;
  unitCost = Costs::hashProbeCost(buildSize) + fanout * rowCost +
      leftKeys.size() * Costs::kHashColumnCost;
}

} // namespace facebook::verax
