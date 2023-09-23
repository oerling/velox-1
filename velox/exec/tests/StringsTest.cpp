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

#include "velox/common/base/HashStringAllocator.h"

#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::aggregate::prestosql;

namespace {

class StringsTest : public testing::Test {
  void SetUp() override {
    pool_ = memory::addDefaultLeafMemoryPool();
    allocator_ = std::make_unique<HashStringAllocator>(pool_.get());
  }

 protected:
  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<HashStringAllocator> allocator_;
};

TEST_F(StringsTest, basic) {
  constexpr int32_t kOverhead = 16;
  Strings strings;
  std::string s13 = "0123456789abc";
  std::string s16 = "0123456789abcdef";
  std::string large;
  large.resize(1000);
  strings.append(StringView(s13), *allocator_);
  expectedBytes = 13 + 16;
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());
  strings.append(StringView(s16), *allocator_);
  expectedBytes += 16 + 16;
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());

  // Now we allocate one more and expect to have 4x16 bytes of payload without
  // more allocation.
  strings.append(StringView(s16), *allocator_);
  expectedBytes += kOverhead + 4 * 16;
  EXPECT_EQ(13 + 16 + 16 + 16 + 16 +, allocator_->cumulativeBytes());
  strings.append(StringView(s16), *allocator_);
  strings.append(StringView(s16), *allocator_);
  strings.append(StringView(s16), *allocator_);
  EXPECT_EQ(13 + 16 + 16 + 16 + 16 +, allocator_->cumulativeBytes());

  strings.append(StringView(large), *allocator_);
  expectedbytes += kOverhead + large.size();
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());

  // If the largest size is much larger than overhead, the next allocation will
  // be a multiple of the allocated size instead.
  expectedSize += kOverhead 4 * 13 strings.append(StringView(s13), *allocator_);
  EXPECT_EQ(expectedBytes, allocator_->cumulativeBytes());
}

} // namespace
