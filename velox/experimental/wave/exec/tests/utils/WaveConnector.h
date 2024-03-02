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

#include "velox/connectors/Connector.h"
#include "velox/experimental/wave/exec/WaveDataSource.h"
#include "velox/experimental/wave/exec/tests/utils/FileFormat.h"
#include "velox/experimental/wave/exec/tests/utils/WaveTestDataSource.h"

namespace facebook::velox::wave::test {

/// Connector, ConnectorFactory and DataSource for Wave memory mock tables.
class WaveMockDataSource : public connector::DataSource {
 public:
  WaveMockDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      const connector::ConnectorQueryCtx* connectorQueryCtx)
      : outputType_(outputType),
        tableHandle_(tableHandle),
        columnHandles_(columnHandles),
        connectorQueryCtx_(connectorQueryCtx) {}

  void addSplit(std::shared_ptr<connector::ConnectorSplit> split) override {
    split_ = std::dynamic_pointer_cast<WaveTestConnectorSplit>(split);
    VELOX_CHECK_NOT_NULL(split_);
  }

  void setFromDataSource(std::unique_ptr<DataSource> sourceUnique) override {
    auto other = dynamic_cast<WaveMockDataSource*>(sourceUnique.get());
    split_ = std::move(other->split_);
  }

  std::shared_ptr<WaveDataSource> toWaveDataSource() override;

  std::optional<RowVectorPtr> next(
      uint64_t size,
      velox::ContinueFuture& future) {
    VELOX_UNSUPPORTED();
  }

  void addDynamicFilter(
      column_index_t outputChannel,
      const std::shared_ptr<common::Filter>& filter) {
    waveDataSource_->addDynamicFilter(outputChannel, filter);
  }

  uint64_t getCompletedBytes() override {
    waveDataSource_->getCompletedBytes();
  }

  uint64_t getCompletedRows() override {
    waveDataSource_->getCompletedBytes();
  };

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    return waveDataSource_->runtimeStats();
  }

 private:
  const RowTypePtr outputType_;
  std::shared_ptr<connector::ConnectorTableHandle> tableHandle_;
  const std::
      unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
          columnHandles_;
  const connector::ConnectorQueryCtx* connectorQueryCtx_;
  std::shared_ptr<WaveTestConnectorSplit> split_;
  std::shared_ptr<WaveTestDataSource> waveDataSource_;
};

class WaveMockConnector : public connector::Connector {
 public:
  WaveMockConnector(
      const std::string& id,
      std::shared_ptr<const Config> config,
      folly::Executor* executor);

  const std::shared_ptr<const Config>& connectorConfig() const override {
    return config_;
  }

  bool canAddDynamicFilter() const override {
    return true;
  }

  std::unique_ptr<connector::DataSource> createDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      connector::ConnectorQueryCtx* connectorQueryCtx) override {
    return std::make_unique<WaveMockDataSource>(
        outputType, tableHandle, columnHandles, connectorQueryCtx);
  }

  bool supportsSplitPreload() override {
    return true;
  }

  std::unique_ptr<connector::DataSink> createDataSink(
      RowTypePtr inputType,
      std::shared_ptr<connector::ConnectorInsertTableHandle>
          connectorInsertTableHandle,
      connector::ConnectorQueryCtx* connectorQueryCtx,
      connector::CommitStrategy commitStrategy) override final {
    VELOX_UNSUPPORTED();
  }

  folly::Executor* FOLLY_NULLABLE executor() const override {
    return executor_;
  }

 protected:
  folly::Executor* FOLLY_NULLABLE executor_;
  std::shared_ptr<const Config> config_;
};

class WaveMockConnectorFactory : public connector::ConnectorFactory {
 public:
  static constexpr const char* kWaveConnectorName = "wave";

  WaveMockConnectorFactory() : ConnectorFactory(kWaveConnectorName) {}

  explicit WaveMockConnectorFactory(const char* FOLLY_NONNULL connectorName)
      : ConnectorFactory(connectorName) {}

  void initialize() override {}

  std::shared_ptr<connector::Connector> newConnector(
      const std::string& id,
      std::shared_ptr<const Config> config,
      folly::Executor* FOLLY_NULLABLE executor = nullptr) override {
    return std::make_shared<WaveMockConnector>(id, config, executor);
  }
};
} // namespace facebook::velox::wave::test
