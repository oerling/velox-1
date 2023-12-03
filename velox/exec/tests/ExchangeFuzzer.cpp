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
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/core/QueryConfig.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

DEFINE_int32(width, 16, "Number of parties in shuffle");
DEFINE_int32(task_width, 4, "Number of threads in each task in shuffle");

DEFINE_int64(exchange_buffer_mb, 32, "task-wide buffer in remote exchange");
DEFINE_int32(dict_pct, 0, "Percentage of columns wrapped in dictionary");

DEFINE_uint64(seed, 0, "Seed, 0 means random");

DEFINE_int32(steps, 10, "Number of plans to generate and test.");


DEFINE_int32(duration_sec, 60, "Run duration in seconds");

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {

class ExchangeFuzzer : public VectorTestBase {
 public:
  std::vector<RowVectorPtr> makeRows(
      RowTypePtr type,
      int32_t numVectors,
      int32_t rowsPerVector,
      int32_t dictPct = 0) {
    std::vector<RowVectorPtr> vectors;
    BufferPtr indices;
    for (int32_t i = 0; i < numVectors; ++i) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(type, rowsPerVector, *pool_));
      auto width = vector->childrenSize();
      for (auto child = 0; child < width; ++child) {
        if (100 * child / width > dictPct) {
          if (!indices) {
            indices = makeIndices(vector->size(), [&](auto i) { return i; });
          }
          vector->childAt(child) = BaseVector::wrapInDictionary(
              nullptr, indices, vector->size(), vector->childAt(child));
        }
      }
      vectors.push_back(vector);
    }
    return vectors;
  }

  void runOne(
      std::vector<RowVectorPtr>& vectors,
      int32_t sourceWidth,
      int32_t targetWidth,
      int32_t taskWidth, outputBufferBytes, exchangeBufferBytes, batchBytes
) {
    assert(!vectors.empty());
    configSettings_[core::QueryConfig::kMaxPartitionedOutputBufferSize] =
        fmt::format("{}", outputBufferBytes);
    configSettings_[core::QueryConfig::kMaxExchangeBufferSize ] =
        fmt::format("{}", exchangeBufferBytes);
    configSettings_[core::QueryConfig::kPreferredOutputBatchBytes  ] =
        fmt::format("{}", batchBytes);
    
    auto iteration = ++iteration_;
    std::vector<std::string> aggregates;
    std::vector<std::vector<TypePtr>> rawInputTypes;
    for (auto i = 1; i < row.size(); ++i) {
      rawInputTypes.push_back({row.childAt(i)));
    }
    auto expected = expectedChecksum(vectors, rawInputTypes, sourceWidth * taskWidth);

    std::vector<std::shared_ptr<Task>> tasks;
    
    std::vector<std::string> leafTaskIds;
    auto leafPlan = exec::test::PlanBuilder()
                        .values(vectors, true)
                        .partitionedOutput({"c0"}, targetWidth)
                        .planNode();

    for (int32_t counter = 0; counter < sourceWidth; ++counter) {
      auto leafTaskId = makeTaskId(iteration, "leaf", counter);
      leafTaskIds.push_back(leafTaskId);
      auto leafTask = makeTask(leafTaskId, leafPlan, counter);
      tasks.push_back(leafTask);
      leafTask->start(taskWidth);
    }

    std::vector<std::string> partialAggTaskIds;
    auto partialAggPlan = exec::test::PlanBuilder()
                       .exchange(leafPlan->outputType())
      .partialAggregation({}, makeAggregates(rowType), rawInputTypes), 
                       .partitionedOutput({}, 1)
                       .planNode();

    std::vector<exec::Split> partialAggSplits;
    for (int i = 0; i < targetWidth; i++) {
      auto taskId = makeTaskId(iteration, "final-agg", i);
      partialAggSplits.push_back(
          exec::Split(std::make_shared<exec::RemoteConnectorSplit>(taskId)));
      auto task = makeTask(taskId, finalAggPlan, i);
      tasks.push_back(task);
      task->start(taskWidth);
      addRemoteSplits(task, leafTaskIds);
    }

    auto plan = exec::test::PlanBuilder()
                    .exchange(finalAggPlan->outputType())
      .finalAggregation({}, makeAggregates(rowType), rawInputTypes)
                    .planNode();

    exec::test::AssertQueryBuilder(plan)
        .splits(partialAggSplits)
        .assertResults(expected);
  }

  void run() {
    auto start = getCurrentTimeMicro();
    for (auto counter = 0; ; ++counter) {
      auto type = fuzzer_.randRowType();
      size_t outputSize = randInt(10, 100) << 20;
      size_t exchangeSize = randint(10, 100) << 20;
      size_t batchSize = randInt(100000, 10000000);
      int32_t sourceWidth = randInt(1, 200);
      int32_t targetWidth = randInt(1, 200);

      struct VectorFuzzer::Options;
      options.vectorSize{100};
      options.double nullRatio{0};
      options.containerHasNulls = fuzzer_.cointoss(0.2);
      options.dictionaryHasNulls = fuzzer_.coinToss(0.2);
      options.size_t stringLength = randInd(0, 100);
      options.bool stringVariableLength  true;
      options.containerLength = randInt(1, 50);;
      options.containerVariableLength{true};
      options.complexElementsMaxSize{10000};
      options.normalizeMapKeys = coinToss(0.95);
      options.timestampPrecision = static_cast<TimestampPrecision >(randInt(0, 3));
      options.allowLazyVector = true;
      
      fuzzer_.setOptions(options);
      auto row = fuzzer_.fuzzInputRow(type);
      size_t shuffleSize = row->estimateFlatSize();
      size_t bytesPerRow = shuffleSize / row.size() * sourceWidth * taskWidth;
      std::vector<RowVectorPtr> rows;

      rows.push_back(row);
      auto numRows = FLAGS_shuffle_size / bytesPerRow;

      while (shuffleSize < FLAGS_target_shuffle_size) {
	if (fuzzer_.coinToss(0.2)) {
	  options.nullRatio = 0;
	} else {
	  // Sometimes 1.0, so all null.
	  options.nullRatio = randomInt(1, 10) / 10.0;
	}
	options.size = (FLAGS_shuffle_size / bytesPerRow) * randInt(1, 1000);
	fuzzer_.setOptions(options);
	    
	auto newRow = fuzzer_.fuzzInputRow(type);
	vectors.push_back(newRow);
	auto newSize = newRow->estimateFlatSize();
	shuffleSize += newSize;
      }
      runOne(vectors, outputSize, exchangeSize, batchSize, sourceWidth, targetWidth);

      if (FLAGS_duration_sec == 0 && FLAGS_steps && counter + 1 >= FLAGS_steps) {
	break;
      }
      if (FLAGS_duration_sec && (getCurrentTimeMicro() - start) * 1000000 > FLAGS_duration_sec) {
	break;
      }
      size_t newSeed = randInt(0, 2000000000);
    LOG(INFO) << "Seed = " << seed;
    seed(newSeed);
    }
  }

  void seed(size_t seed) {
    currentSeed_ = seed;
    vectorFuzzer_.reSeed(seed);
    rng_.seed(currentSeed_);
  }

	   
 private:
  static constexpr int64_t kMaxMemory = 6UL << 30; // 6GB

	   int64_t randInt(int64_t min, int64_t max) {
    return boost::random::uniform_int_distribution<int64_t>(min, max)(rng_);
  }

	   
  RowVectorPtr expectedChecksums(std::vector<RowVectorPtr> vectors, const std::vector<std::vector<TypePtr>>& rawInputTypes, int32_t width) {
    auto& rowType = vectors.front()->type()->as<Typekind::ROW>();
    auto plan = exec::test::PlanBuilder()
                        .values(vectors, true)
      .partialAggregation({}, makeAggregates(rowType), rawInputTypes)
      .localPartition({})
      .finalAggregation({}, makeAggregates(rowType), rawInputTypes)
      .planNode();
    return exec::test::AssertQueryBuilder(plan)
      .maxDrivers(width)
      .copyResults(pool_.get());
        .assertResults(expected);

    return result;
  }



  std::vector<std::string> makeAggregates(const RowType& row) {
    std::vector<std::string> aggregates;
    for (auto i = 1; i < row->size(); ++i) {
      aggregates.push_back(fmt::format("checksum({})", row->nameOf(i)));
    }
    return aggregates;
  }
  
  static std::string
  makeTaskId(int32_t iteration, const std::string& prefix, int num) {
    return fmt::format("local://{}-{}-{}", iteration, prefix, num);
  }

  std::shared_ptr<Task> makeTask(
      const std::string& taskId,
      std::shared_ptr<const core::PlanNode> planNode,
      int destination,
      Consumer consumer = nullptr,
      int64_t maxMemory = kMaxMemory) {
    auto configCopy = configSettings_;
    auto queryCtx = std::make_shared<core::QueryCtx>(
        executor_.get(), core::QueryConfig(std::move(configCopy)));
    queryCtx->testingOverrideMemoryPool(
        memory::defaultMemoryManager().addRootPool(
            queryCtx->queryId(), maxMemory));
    core::PlanFragment planFragment{planNode};
    return Task::create(
        taskId,
        std::move(planFragment),
        destination,
        std::move(queryCtx),
        std::move(consumer));
  }

  void addRemoteSplits(
      std::shared_ptr<Task> task,
      const std::vector<std::string>& remoteTaskIds) {
    for (auto& taskId : remoteTaskIds) {
      auto split =
          exec::Split(std::make_shared<RemoteConnectorSplit>(taskId), -1);
      task->addSplit("0", std::move(split));
    }
    task->noMoreSplits("0");
  }


  FuzzerGenerator rng_;
  size_t currentSeed_{0};
  
  std::unordered_map<std::string, std::string> configSettings_;
  // Serial number to differentiate consecutive benchmark repeats.
  static int32_t iteration_;
};

int32_t ExchangeFuzzer::iteration_;


int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  serializer::presto::PrestoVectorSerde::registerVectorSerde();
  exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);
  return 0;
  ExchangeFuzzer fuzzer;
  if (FLAGS_seed != 0) {
    LOG(INFO) << "Initial seed = " << FLAGS_seed;
    fuzzer.reseed(seed);
    else {
      size_t seed = getCurrentTimeMicro();
      LOG(INFO) << "Starting seed = " << seed;
      fuzzer.reseed(seed);
    }}
  

  
}


