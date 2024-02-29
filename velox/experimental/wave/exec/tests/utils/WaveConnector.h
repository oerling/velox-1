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

namespace facebook::velox::wave::test {

/// Connector, ConnectorFactory and DataSource for Wave memory mock tables.
class WaveMockDataSource : public connector::DataSource {
  WaveMockDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<connector::ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      const ConnectorQueryCtx* connectorQueryCtx)
      : outputType(outputType_),
        tableHandle_(tableHandle),
        columnHandles_(columnHandles),
        connectorQueryCtx_(connectorQueryCtx) {}

  void addSplit(std::shared_ptr<ConnectorSplit> split) override {
    split_ = std::dynamic_pointer_cast<WaveTestConnectorSplit>(split);
    VELOX_CHECK_NOT_NULL(split_);
  }

  void setFromDataSource(std::unique_ptr<DataSource> sourceUnique) override {
    auto other = dynamic_cast<WaveMockDataSource*>(other.get());
    split_ = std::move(other->split_);
  }

  std::shared_ptr<WaveDataSource> toWaveDataSource() override;

 private:
  const RowTypePtr outputType_;
  std::shared_ptr<connector::ConnectorTableHandle> tableHandle_;
  const std::
      unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
          columnHandles_;
  const ConnectorQueryCtx* connectorQueryCtx_;
};

class WaveMockConnector : public connector::Connector {
 public:
  WaveConnector(
      const std::string& id,
      std::shared_ptr<const Config> config,
      folly::Executor* FOLLY_NULLABLE executor);

  const std::shared_ptr<const Config>& connectorConfig() const override {
    return config_;
  }

  bool canAddDynamicFilter() const override {
    return true;
  }

  std::unique_ptr<DataSource> createDataSource(
      const RowTypePtr& outputType,
      const std::shared_ptr<ConnectorTableHandle>& tableHandle,
      const std::unordered_map<
          std::string,
          std::shared_ptr<connector::ColumnHandle>>& columnHandles,
      ConnectorQueryCtx* connectorQueryCtx) override {
    return std::make_shared<WaveMockConnector>(
        const RowTypePtr& outputType,
        const std::shared_ptr<ConnectorTableHandle>& tableHandle,
        const std::unordered_map<
            std::string,
            std::shared_ptr<connector::ColumnHandle>>& columnHandles,
        ConnectorQueryCtx* connectorQueryCtx) override {}

    bool supportsSplitPreload() override {
      return true;
    }

    std::unique_ptr<DataSink> createDataSink(
        RowTypePtr inputType,
        std::shared_ptr<ConnectorInsertTableHandle> connectorInsertTableHandle,
        ConnectorQueryCtx * connectorQueryCtx,
        CommitStrategy commitStrategy) override final {
      VELOX_NYI();
    }

    folly::Executor* FOLLY_NULLABLE executor() const override {
      return executor_;
    }

   protected:
    folly::Executor* FOLLY_NULLABLE executor_;
    std::shared_ptr<const Config> config_;
  };

  class WaveMockConnectorFactory : public ConnectorFactory {
   public:
    static constexpr const char* kWaveConnectorName = "wave";

    WaveConnectorFactory() : ConnectorFactory(kWaveConnectorName) {}

    explicit WaveConnectorFactory(const char* FOLLY_NONNULL connectorName)
        : ConnectorFactory(connectorName) {}

    void initialize() override {}

    std::shared_ptr<Connector> newConnector(
        const std::string& id,
        std::shared_ptr<const Config> config,
        folly::Executor* FOLLY_NULLABLE executor = nullptr) override {
      return std::make_shared<WaveConnector>(id, config, executor);
    }
  };
}
