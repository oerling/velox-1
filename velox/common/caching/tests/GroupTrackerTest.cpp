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

struct TestGroup {
  std::string name;
  std::vector<std::string> files;
  std::vector<int32_t> columnSizes; 
};

class GroupTrackerTest : public testing::Test {
 protected:

  void SetUp() override {
    rng_.seed(1);
  }
  
  void makeGroups(int numGroups) {
    for (auto i = 0; i < numGroups; ++i) {
      TestGroup group;
      int32_t sizeClass = random(100);
      int32_t numColumns = 0;
      if (sizeClass < 30) {
	numColumns = 15 + random(10);
      }
    }
  }

  void run(int32_t numGroups, int32_t numFiles, int32_t numColumns) {

  }


  
  std::string groupName(const std::string filename) {
    const char* slash = strrchr(name.c_str(), '/');
    if (slash) {
      return std::string(filename, slash -str.data());
    }
    return filename;
  }

  bool shouldReference(uint64_t fileId, uint64_t groupId, int32_t columnId) {
  }

  bool shouldRead(uint64_t groupId, int32_t column) {
  }
  

  int32_t  random(int32_t range, int32_t maskRange) {
    return (folly::Random::rand32(rng_) % range) |
      (folly::Random::rand32(rng_) %  maskRange);
  }

    int32_t  random(int32_t range) {
      return folly::Random::rand32(rng_) % range;
  }

    std::vector<int32_t> columnSize_{1000, 2000, 3000,3000,3000, 4000, 10000, 15000, 100000};


  folly::Random::DefaultGenerator rng_;

};

TEST_F(GroupTrackerTest, biased) {
  auto& stats = GroupStats::instance();
  for (auto i = 0; i < kQueries; ++i) {
    run(
  }
}


