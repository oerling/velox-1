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

#include "velox/exec/tests/utils/Cursor.h"
#include "velox/experimental/query/ExecutablePlan.h"

namespace facebook::velox::exec {
  
class LocalRunner {
 public:
  LocalRunner(
      std::vector<ExecutableFragment> plan,
      std::shared_ptr<core::QueryCtx> queryCtx,
      int32_t numDrivers)
    : 
        numDrivers_(numDrivers),
        plan_(std::move(plan)) {
    params_.queryCtx = queryCtx;
  }
	LocalRunner(const std::string& text, verax::SchemaPrtr schema, StatsSource* stats);
	
  test::TaskCursor*
cursor();

private:
test::CursorParameters params_;
const int32_t numDrivers_;
std::vector<ExecutableFragment> plan_;
std::unique_ptr<test::TaskCursor> cursor_;
std::vector<std::vector<std::shared_ptr<Task>>> stages_;
};

} // namespace facebook::velox::exec
