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

#include "velox/common/base/SimdUtil.h"
#include "velox/core/PlanNode.h"
#include "velox/experimental/query/QueryGraph.h"

namespace facebook::velox::query {

/// Instance of query optimization. Comverts a plan and schema into an optimized
/// plan. Depends on QueryGraphContext being set on the calling thread.
class Optimization {
 public:
  Optimization(
      const std::shared_ptr<core::PlanNode>& plan,
      const Schema& schema);

 private:
  DerivedTablePtr makeQueryGraph();

  const Schema& schema_;
  std::shared_ptr<core::PlanNode> inputPlan_;
  DerivedTablePtr root_;
};

} // namespace facebook::velox::query
