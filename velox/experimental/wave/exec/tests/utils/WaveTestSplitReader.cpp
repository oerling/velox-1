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

  WaveTestSplitReader::WaveTestSplitReader(const std::shared_ptr<connector::ConnectorSplit>& split,
					   const SplitReaderParams& params) {
    stripe_ = test::Table::getStripe(split->path);
  }

int32_t WaveTestDataSource::canAdvance() {
  return 0;
}

void WaveTestDataSource::schedule(WaveStream& stream, int32_t maxRows) {
  VELOX_NYI();
}

vector_size_t WaveTestDataSource::outputSize(WaveStream& stream) const {
  return 0;
}

bool WaveTestDataSource::isFinished() const {
  return false;
}
  namespace {
  class WaveTestSplitReaderFactory {
  public:
    std::unique_ptr<WaveSplitReader> create (const std::shared_ptr<HiveConnectorSplit>& split,
					     const SplitReaderParams& params) override {
      if (memcmp(split.path.data(), "wavemock://", 11) == 0) {
	return std::make_unique<WaveTestSplitReader>(split, params);
      }
      return nullptr;
    }

  };
  }
    
  static void WaveTestSplitReader::registerTestSplitReader() {
    WaveSplitReader::register(std::make_unique<WaveTestSplitReaderFactory>());
  }

  
} // namespace facebook::velox::wave::test
