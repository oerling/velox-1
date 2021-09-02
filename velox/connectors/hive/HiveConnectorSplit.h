/*
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

#include <optional>
#include <unordered_map>
#include "velox/connectors/Connector.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::connector::hive {

const std::string kHiveConnectorName = "hive";

struct HiveConnectorSplit : public connector::ConnectorSplit {
  const std::string filePath;
  const uint64_t start;
  const uint64_t length;
  const std::unordered_map<std::string, std::string> partitionKeys;
  std::optional<int32_t> tableBucketNumber;

  // Serializes access to prefetch members.
  std::mutex prefetchMutex;

  // True if an async prefetch thread or the table scan thread have started
  // processsing. If true, the async tread will do nothing and the table scan
  // thred will wait for the preheat to finish.
  bool prefetchInProgress{false};

  // Allows the table scan thread to get a future to wait for preheat to finish.
  folly::Promise<bool> prefetchPromise;

  // Set by the async prefetch threads when it completes the prefetching.
  std::shared_ptr<DataSource> prefetchedDataSource;

  HiveConnectorSplit(
      const std::string& connectorId,
      const std::string& _filePath,
      uint64_t _start = 0,
      uint64_t _length = std::numeric_limits<uint64_t>::max(),
      const std::unordered_map<std::string, std::string>& _partitionKeys = {},
      std::optional<int32_t> _tableBucketNumber = std::nullopt)
      : ConnectorSplit(connectorId),
        filePath(_filePath),
        start(_start),
        length(_length),
        partitionKeys(_partitionKeys),
        tableBucketNumber(_tableBucketNumber) {}

  std::string toString() const override {
    if (tableBucketNumber.has_value()) {
      return fmt::format(
          "[file {} {} - {} {}]",
          filePath,
          start,
          length,
          tableBucketNumber.value());
    }
    return fmt::format("[file {} {} - {}]", filePath, start, length);
  }
};

} // namespace facebook::velox::connector::hive
