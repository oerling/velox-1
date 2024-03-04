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

#include "velox/experimental/wave/exec/tests/utils/WaveTestSplitReader.h"

namespace facebook::velox::wave::test {

WaveTestSplitReader::WaveTestSplitReader(
    const std::shared_ptr<connector::ConnectorSplit>& split,
    const SplitReaderParams& params) {
  auto hiveSplit =
      dynamic_cast<connector::hive::HiveConnectorSplit*>(split.get());
  VELOX_CHECK_NOT_NULL(hiveSplit);
  stripe_ = test::Table::getStripe(hiveSplit->filePath);
  VELOX_CHECK_NOT_NULL(stripe_);
}

int32_t WaveTestSplitReader::canAdvance() {
  return 0;
}

void WaveTestSplitReader::schedule(WaveStream& stream, int32_t maxRows) {
  VELOX_NYI();
}

vector_size_t WaveTestSplitReader::outputSize(WaveStream& stream) const {
  return 0;
}

bool WaveTestSplitReader::isFinished() const {
  return false;
}
namespace {
class WaveTestSplitReaderFactory : public WaveSplitReaderFactory {
 public:
  std::unique_ptr<WaveSplitReader> create(
      const std::shared_ptr<connector::ConnectorSplit>& split,
      const SplitReaderParams& params) override {
    auto hiveSplit =
        dynamic_cast<connector::hive::HiveConnectorSplit*>(split.get());
    if (!hiveSplit) {
      return nullptr;
    }
    if (hiveSplit->filePath.size() > 11 &&
        memcmp(hiveSplit->filePath.data(), "wavemock://", 11) == 0) {
      return std::make_unique<WaveTestSplitReader>(split, params);
    }
    return nullptr;
  }
};
} // namespace

//  static
void WaveTestSplitReader::registerTestSplitReader() {
  WaveSplitReader::registerFactory(
      std::make_unique<WaveTestSplitReaderFactory>());
}

} // namespace facebook::velox::wave::test
