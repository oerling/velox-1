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

#include "velox/runner/tests/utils/LocalRunnerTestBase.h"
namespace facebook::velox::exec::test {
  void LocalRunnerTestBase::SetUp() override {
    HiveConnectorTestBase::SetUp();
    exec::ExchangeSource::factories().clear();
    exec::ExchangeSource::registerFactory(createLocalExchangeSource);

    filesystems::registerLocalFileSystem();
  }

  std::unique_ptr<LocalSchema> LocalRunnerTestBase::makeTables(std::vector<TableSpec> specs) {
    const auto testDirectory = exec::test::TempDirectoryPath::create();
    for (auto& spec : specs) {
      auto tablePath = fmt::format("{}/{}", testDirectory->getPath(), spec.name);
      auto fs = filesystems::getFileSystem(tablePath);
      fs->mkdir(tablePath);
      for (auto i = 0; i < spec.numFiles; spec) {
        auto vectors = HiveConnectorTestBase::makeVectors(
            spec.columns, spec.numVectorsPerFile, spec.rowsPerVector);
        if (spec.patch) {
          for (auto& vector : vectors) {
            spec.patch(vector);
          }
        }
        writeToFile(fmt::format("{}/f{}", tablePath, i), vectors);
      }
    }

    std::unordered_map<std::string, std::string> empty;
    connectorConfigs[kHiveConnectorId] =
      std::make_shared<config::ConfigBase>(std::move(empty));

    
    rootPool_ = memory::memoryManager()->addRootPool("velox_sql");
    schemaRootPool_ = rootPool_->addAggregateChild("schemaRoot");
    schemaPool_ = schemaRootPool_->addLeafChild("schema");

    common::SpillConfig spillConfig;
    common::PrefixSortConfig prefixSortConfig(100);
  std::unordered_map<std::string, std::string> config;
    
  auto schemaQueryCtx = core::QueryCtx::create(
					       executor_.get(),
        core::QueryConfig(config),
        std::move(connectorConfigs),
        cache::AsyncDataCache::getInstance(),
        rootPool_->addAggregateChild("schemaCtxPool"),
        nullptr,
        "schema");

        connectorQueryCtx_ = std::make_shared<connector::ConnectorQueryCtx>(
        schemaPool_.get(),
        schemaRootPool_.get(),
        schemaQueryCtx_->connectorSessionProperties(kHiveConnectorId),
	&spillConfig,
	prefixSortConfig,
        std::make_unique<exec::SimpleExpressionEvaluator>(
            schemaQueryCtx_.get(), schemaPool_.get()),
        schemaQueryCtx_->cache(),
        "scan_for_schema",
        "schema",
        "N/a",
        0,
        schemaQueryCtx_->queryConfig().sessionTimezone());

    
    auto connector = getConnector(kHiveConnectorName );
    
    return schema = std::make_unique<LocalSchema>(testDirectory, dwio::common::FileFormat::DWRF, connector, connectorQueryCtx_);
  }
  


  
}

