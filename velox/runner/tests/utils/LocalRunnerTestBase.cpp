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
  void LocalRunnerTestBase::SetUp() {
    HiveConnectorTestBase::SetUp();
    exec::ExchangeSource::factories().clear();
    exec::ExchangeSource::registerFactory(createLocalExchangeSource);

    filesystems::registerLocalFileSystem();
  }

  std::unique_ptr<LocalSchema> LocalRunnerTestBase::makeTables(std::vector<TableSpec> specs, std::shared_ptr<TempDirectoryPath>& directory) {
    directory = exec::test::TempDirectoryPath::create();
    for (auto& spec : specs) {
      auto tablePath = fmt::format("{}/{}", directory->getPath(), spec.name);
      auto fs = filesystems::getFileSystem(tablePath, {});
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

        std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>> connectorConfigs;
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

  auto connectorQueryCtx = std::make_shared<connector::ConnectorQueryCtx>(
        schemaPool_.get(),
        schemaRootPool_.get(),
        schemaQueryCtx->connectorSessionProperties(kHiveConnectorId),
	&spillConfig,
	prefixSortConfig,
        std::make_unique<exec::SimpleExpressionEvaluator>(
            schemaQueryCtx.get(), schemaPool_.get()),
        schemaQueryCtx->cache(),
        "scan_for_schema",
        "schema",
        "N/a",
        0,
        schemaQueryCtx->queryConfig().sessionTimezone());

    
  auto connector = connector::getConnector(kHiveConnectorId );
    
  return std::make_unique<LocalSchema>(directory->getPath(), dwio::common::FileFormat::DWRF, reinterpret_cast<velox::connector::hive::HiveConnector*>(connector.get()), connectorQueryCtx);
  }
  


  
}

