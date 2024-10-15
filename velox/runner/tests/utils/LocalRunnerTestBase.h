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

#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/exec/ExchangeSource.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"

namespace facebook::velox::exec::test {

struct TableSpec {
  std::string name;
  RowTypePtr columns;
  int32_t rowsPerVector{10000};
  int32_t numVectorsPerFile{5};
  int32_t numFiles{5};
  std::function<void(const RowVectorPtr& vector)> patch;
};

class LocalRunnerTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {);

  std::unique_ptr<LocalSchema> makeTables(std::vector<TableSpec> specs);

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> schemaPool_;
  std::shared_ptr<memory::MemoryPool> schemaRootPool_;


};

} // namespace facebook::velox::exec::test
