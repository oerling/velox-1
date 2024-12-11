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

#include "velox/common/base/Fs.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/connectors/Connector.h"

namespace facebook::velox::runner {

/// Base class for collection of tables. A query executes against a
/// Schema The schema is mutable and may acquire tables and the
/// tables may acquire stats during their lifetime.
class Schema {
 public:
  virtual ~Schema() = default;

  Schema(const std::string& name, memory::MemoryPool* pool)
      : name_(name), pool_(std::move(pool)) {}

  virtual const conector::Table* findTable(const std::string& name);
};

} // namespace facebook::velox::runner
