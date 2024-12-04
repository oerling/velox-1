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

#include "velox/experimental/query/QueryGraphContext.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"

namespace facebook::velox::optimizer {

QueryGraphContext*& queryCtx() {
  thread_local QueryGraphContext* context;
  return context;
}

PlanObjectP QueryGraphContext::dedup(PlanObjectP object) {
  auto pair = deduppedObjects_.insert(object);
  return *pair.first;
}

const char* QueryGraphContext::toName(std::string_view str) {
  auto it = names_.find(str);
  if (it != names_.end()) {
    return it->data();
  }
  char* data = allocator_.allocate(str.size() + 1)->begin(); // NOLINT
  memcpy(data, str.data(), str.size());
  data[str.size()] = 0;
  names_.insert(std::string_view(data, str.size()));
  return data;
}

Name toName(std::string_view string) {
  return queryCtx()->toName(string);
}

} // namespace facebook::velox::optimizer
