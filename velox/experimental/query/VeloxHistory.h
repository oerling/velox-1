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

#include "velox/experimental/query/Plan.h"

namespace facebook::verax {

class VeloxHistory : public History {
 public:
  virtual std::optional<Cost> findCost(RelationOp& op) override {
    return std::nullopt;
  }

  void recordCost(const RelationOp& op, Cost cost) override {}

  bool setLeafSelectivity(BaseTable& table) override;
};

} // namespace facebook::verax
