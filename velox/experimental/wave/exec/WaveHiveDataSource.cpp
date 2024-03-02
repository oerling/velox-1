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

#include "velox/experimental/wave/exec/WaveHiveDataSource.h"

namespace facebook::velox::wave {

// static
void WaveHiveDataSource::register() {
  connector::hive::DataSource::registerWaveDelegateHook(
      [](const std::shared_ptr<HiveTableHandle>& hiveTableHandle,
         const std::shared_ptr<common::ScanSpec>& scanSpec,
         const RowTypePtr& readerOutputType,
         std::unordered_map<std::string, std::shared_ptr<HiveColumnHandle>>*
             partitionKeys,
         FileHandleFactory* fileHandleFactory,
         folly::Executor* executor,
         const ConnectorQueryCtx* connectorQueryCtx,
         const std::shared_ptr<HiveConfig>& hiveConfig,
         const std::shared_ptr<io::IoStatistics>& ioStats,
         const ExprSet& remainingFilter) {
        return std::make_shared<WaveHiveDataSource>(
            hiveTableHandle,
            scanSpec,
            readerOutputType,

            partitionKeys,
            fileHandleFactory,
            executor,
            connectorQueryCtx,
            hiveConfig,
            ioStats,
            remainingFilter)
      });
}

} // namespace facebook::velox::wave
