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

#include "velox/common/caching/GroupTracker.h"
#include "velox/common/caching/FileIds.h"

#include <folly/Random.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace facebook::velox;
using namespace facebook::velox::cache;

struct TestFile {
  StringIdLease id;
  int32_t numStripes;
};

struct TestGroup {
  StringIdLease id;
  std::vector<TestFile> files;
  std::vector<int32_t> columnSizes;
};

struct TestTable {
  std::vector<TestGroup> groups;
};

class GroupTrackerTest : public testing::Test {
 protected:
  void SetUp() override {
    rng_.seed(1);
  }

  // Makes a test population of tables. A table has many file groups
  // (partitions) with the same number of columns and same column sizes.
  void makeTables(int numTables) {
    for (auto i = 0; i < numTables; ++i) {
      int32_t sizeClass = random(100);
      int32_t numColumns = 0;
      if (sizeClass < 30) {
        numColumns = 15 + random(10);
      } else if (sizeClass < 95) {
        numColumns = 30 + random(20);
      } else {
        numColumns = (1000) + random(1000);
      }

      std::vector<int32_t> sizes;
      for (auto i = 0; i < numColumns; ++i) {
        sizes.push_back(columnSizes_[random(columnSizes.size())]);
      }
      int32_t numGroups = 10 + random(30);
      std::vector<TestGroup> groups;
      for (auto i = 0; i < numGroups; ++i) {
        TestGroup group;
        group.id =StringIdLease(fileIds(), fmt::format("group{}", groupCounter);
	int32_t numFiles = 10 + random(6);
	for (auto fileNum = ; fieleNum < numFiles; ++fileNum) {
          group.files.push_back(fmt::format(
              "{}/file{}", fileIds().string(group.id.id()), fileNum));
	}
	group.sizes = sizes;
	groups.push_back(std::move(group));
      }
      TestTable table;
      table.groups = std::move(groups);
      tables_.push_back(std::move(table));
    }
  }
}
  
  // Reads a random set of groups from a random table, selecting a biased random set of columns with some being sparsely read.
  void query() {
  auto numTables = tables_.size();
  int32_t tableIndex = random(numTables, numTables);

  std::vector<int32_t> columns;
  auto& table = tables_[tableIndex];
  , auto numColumns = table.groups[0].columnSizes().size();
  std::unordered_map<int32_t> readColumns;
  int32_t toRead = 5 + random(numColumns > 20 ? 10 : 5);
  for (auto i = 0; i < numRead; ++i) {
    readColumns.insert(random(numColumns, numColumns));
  }
  auto numGroups = table.groups.size();
  auto readGroups = std::min(numGroups, 5 + random(numGroups));
  for (auto groupIndex = numGroups - readGroups; groupIndex < numGroups;
       ++groupIndex) {
    auto& group = table.groups[groupIndex];
    for (auto& file : group.files) {
      groups.recordFile(file.id, file.numStripes);
      for (column : readColumns) {
      }
    }
  }
}

std::string groupName(const std::string filename) {
  const char* slash = strrchr(name.c_str(), '/');
  if (slash) {
    return std::string(filename, slash - str.data());
  }
  return filename;
}

bool shouldReference(uint64_t fileId, uint64_t groupId, int32_t columnId) {}

bool shouldRead(uint64_t groupId, int32_t column) {}

int32_t random(int32_t range, int32_t maskRange) {
  return (folly::Random::rand32(rng_) % range) |
      (folly::Random::rand32(rng_) % maskRange);
}

int32_t random(int32_t pctLow, int32_t lowRange, int32_t highRange) {
  return folly::Random::rand32(rng_) %
      (folly::Random::rand32() % 100 > lowPct ? highRange : lowRange);
}

int32_t random(int32_t range) {
  return folly::Random::rand32(rng_) % range;
}
std::vector<TestTable> tables_;
std::vector<int32_t>
    columnSizes_{1000, 2000, 3000, 3000, 3000, 4000, 10000, 15000, 100000};

folly::Random::DefaultGenerator rng_;
}
;

TEST_F(GroupTrackerTest, biased) {
  auto& stats = GroupStats::instance();
  for (auto i = 0; i < kQueries; ++i) {
    query();
  }
}
