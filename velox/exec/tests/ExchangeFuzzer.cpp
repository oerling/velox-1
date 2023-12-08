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

#include <boost/random/uniform_int_distribution.hpp>
#include <folly/init/Init.h>

#include "velox/common/memory/MmapAllocator.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/VectorSaver.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

#include <fstream>

DEFINE_int32(max_tasks_per_stage, 16, "Max number of sources/destinations");
DEFINE_int32(drivers_per_task, 4, "Number of threads in each task in shuffle");
DEFINE_int64(shuffle_bytes, 4UL << 30, "Shuffle data volume in each step");
DEFINE_int32(max_buffer_mb, 20, "Max buffer size for output/exchange per task");
DEFINE_uint64(seed, 0, "Seed, 0 means random");

DEFINE_int32(steps, 10, "Number of plans to generate and test.");

DEFINE_int32(duration_sec, 0, "Run duration in seconds");

DEFINE_bool(inject_failure, false, "Inject a failure for testing repro");

DEFINE_string(
    repro_path,
    ".",
    "Path for writing repro files in case of failure");

DEFINE_string(
    replay,
    "",
    "File to replay. Files are produced on failure in --repro_path");

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

class ExchangeFuzzer : public VectorTestBase {
 public:
  ExchangeFuzzer() : fuzzer_(options_, pool_.get()) {}

  bool runOne(
      std::vector<RowVectorPtr>& vectors,
      int32_t numSourceTasks,
      int32_t numDestinationTasks,
      int32_t numDriversPerTask,
      int32_t outputBufferBytes,
      int32_t exchangeBufferBytes,
      int32_t batchBytes) {
    assert(!vectors.empty());
    auto iteration = ++iteration_;
    LOG(INFO) << "Iteration " << iteration << " shuffl " << numSourceTasks
              << "x" << numDestinationTasks << " drivers=" << numDriversPerTask
              << " Type " << vectors.front()->type()->toString()
              << " output buffer=" << outputBufferBytes
              << " exchange buffer=" << exchangeBufferBytes
              << " target batch=" << batchBytes;
    configSettings_[core::QueryConfig::kMaxPartitionedOutputBufferSize] =
        fmt::format("{}", outputBufferBytes);
    configSettings_[core::QueryConfig::kMaxExchangeBufferSize] =
        fmt::format("{}", exchangeBufferBytes);
    configSettings_[core::QueryConfig::kPreferredOutputBatchBytes] =
        fmt::format("{}", batchBytes);

    auto& rowType = vectors.front()->type()->as<TypeKind::ROW>();
    std::vector<std::string> aggregates;
    std::vector<std::vector<TypePtr>> rawInputTypes;
    for (auto i = 1; i < rowType.size(); ++i) {
      rawInputTypes.push_back({rowType.childAt(i)});
    }
    auto expected = expectedChecksums(
        vectors, rawInputTypes, numSourceTasks * numDriversPerTask);

    std::vector<std::shared_ptr<Task>> tasks;

    std::vector<std::string> leafTaskIds;
    auto leafPlan = exec::test::PlanBuilder()
                        .values(vectors, true)
                        .partitionedOutput({"c0"}, numDestinationTasks)
                        .planNode();

    for (int32_t counter = 0; counter < numSourceTasks; ++counter) {
      auto leafTaskId = makeTaskId(iteration, "leaf", counter);
      leafTaskIds.push_back(leafTaskId);
      auto leafTask = makeTask(leafTaskId, leafPlan, counter);
      tasks.push_back(leafTask);
      leafTask->start(numDriversPerTask);
    }

    std::vector<std::string> partialAggTaskIds;
    auto partialAggPlan =
        exec::test::PlanBuilder()
            .exchange(leafPlan->outputType())
            .partialAggregation({}, makeAggregates(rowType, 1))
            .partitionedOutput({}, 1)
            .planNode();

    std::vector<exec::Split> partialAggSplits;
    for (int i = 0; i < numDestinationTasks; i++) {
      auto taskId = makeTaskId(iteration, "partial-agg", i);
      partialAggSplits.push_back(
          exec::Split(std::make_shared<exec::RemoteConnectorSplit>(taskId)));
      auto task = makeTask(taskId, partialAggPlan, i);
      tasks.push_back(task);
      task->start(numDriversPerTask);
      addRemoteSplits(task, leafTaskIds);
    }

    auto plan = exec::test::PlanBuilder()
                    .exchange(partialAggPlan->outputType())
                    .finalAggregation(
                        {},
                        makeAggregates(*partialAggPlan->outputType(), 0),
                        rawInputTypes)
                    .planNode();

    try {
      exec::test::AssertQueryBuilder(plan)
          .splits(partialAggSplits)
          .assertResults(expected);
      if (FLAGS_inject_failure) {
        VELOX_FAIL("Testing error");
      }
    } catch (const std::exception& e) {
      LOG(ERROR) << "Terminating with error: " << e.what();
      if (!FLAGS_replay.empty()) {
        LOG(INFO) << "No replay saved since --replay is specified";
        return false;
      }
      saveRepro(
          vectors,
          numSourceTasks,
          numDestinationTasks,
          numDriversPerTask,
          outputBufferBytes,
          exchangeBufferBytes,
          batchBytes);
      return false;
    }
    return true;
  }

  void run() {
    auto start = getCurrentTimeMicro();
    for (auto counter = 0;; ++counter) {
      auto type = fuzzer_.randRowType();
      // Add a bigint c0 partition key column in front
      auto types = type->children();
      auto names = type->names();
      std::vector<TypePtr> allTypes = {BIGINT()};
      std::vector<std::string> allNames = {"c0"};
      allTypes.insert(allTypes.end(), types.begin(), types.end());
      allNames.insert(allNames.end(), names.begin(), names.end());
      auto rowType = ROW(std::move(allNames), std::move(allTypes));
      size_t outputSize = randInt(4, std::max(5, FLAGS_max_buffer_mb)) << 20;
      size_t exchangeSize = randInt(4, std::max(5, FLAGS_max_buffer_mb)) << 20;
      size_t batchSize = randInt(100000, 10000000);
      int32_t numSourceTasks = randInt(1, FLAGS_max_tasks_per_stage);
      int32_t numDestinationTasks = randInt(2, FLAGS_max_tasks_per_stage);

      options_.vectorSize = 100;
      options_.nullRatio = 0;
      options_.containerHasNulls = fuzzer_.coinToss(0.2);
      options_.dictionaryHasNulls = false;
      // TODO: fuzzer_.coinToss(0.2); This does not work because
      // toElementRows with null adding dicts ignores nulls added by
      // dicts..
      options_.stringLength = randInt(1, 100);
      options_.stringVariableLength = true;
      options_.containerLength = randInt(1, 50);
      options_.containerVariableLength = true;
      options_.complexElementsMaxSize = 20000;
      options_.maxConstantContainerSize = 2;
      options_.normalizeMapKeys = fuzzer_.coinToss(0.95);
      options_.timestampPrecision =
          static_cast<VectorFuzzer::Options::TimestampPrecision>(randInt(0, 3));
      options_.allowLazyVector = false;

      fuzzer_.setOptions(options_);

      // We make a first vector of 100 elements to see the bytes per
      // row. We then make as many vectors as it takes to reach a
      // shuffle volume of --shuffle_bytes. Each row is produced once
      // by each Driver in each task of the source
      // stage. 'shuffledBytesPerRow' is the row size times source
      // stage task count times drivers per task.
      auto row = fuzzer_.fuzzInputRow(rowType);
      size_t shuffleBytes = row->estimateFlatSize();
      size_t shufledBytesPerRow = 20 +
          (shuffleBytes / row->size() * numSourceTasks *
           FLAGS_drivers_per_task);
      std::vector<RowVectorPtr> vectors;

      vectors.push_back(row);
      auto maxBatch = std::min<int32_t>(
          10000,
          std::max<int64_t>(10, FLAGS_shuffle_bytes / shufledBytesPerRow));

      while (shuffleBytes < FLAGS_shuffle_bytes) {
        if (fuzzer_.coinToss(0.2)) {
          options_.nullRatio = 0;
        } else {
          // Sometimes 1.0, so all null.
          options_.nullRatio = randInt(1, 10) / 10.0;
        }
        options_.vectorSize = randInt(1, maxBatch);
        fuzzer_.setOptions(options_);

        auto newRow = fuzzer_.fuzzInputRow(rowType);
        vectors.push_back(newRow);
        auto newSize = newRow->estimateFlatSize() * numSourceTasks *
            FLAGS_drivers_per_task;
        shuffleBytes += newSize;
      }

      if (!runOne(
              vectors,
              numSourceTasks,
              numDestinationTasks,
              FLAGS_drivers_per_task,
              outputSize,
              exchangeSize,
              batchSize)) {
        LOG(INFO) << "Terminating with error";
        exit(1);
      }
      LOG(INFO) << "Memory after run="
                << succinctBytes(memory::AllocationTraits::pageBytes(
                       memory::MemoryManager::getInstance()
                           .allocator()
                           .numAllocated()));

      if (FLAGS_duration_sec == 0 && FLAGS_steps &&
          counter + 1 >= FLAGS_steps) {
        break;
      }
      if (FLAGS_duration_sec &&
          (getCurrentTimeMicro() - start) / 1000000 > FLAGS_duration_sec) {
        break;
      }
      size_t newSeed = randInt(0, 2000000000);
      LOG(INFO) << "Seed = " << newSeed;
      seed(newSeed);
    }
    LOG(INFO) << "Finishing after " << iteration_ << " cases, "
              << (getCurrentTimeMicro() - start) / 1000000 << "s";
  }

  bool replay() {
    std::ifstream in(FLAGS_replay);
    int32_t numSourceTasks;
    int32_t numDestinationTasks;
    int32_t numDriversPerTask;
    int32_t outputBufferBytes;
    int32_t exchangeBufferBytes;
    int32_t batchBytes;
    in >> numSourceTasks;
    in >> numDestinationTasks;
    in >> numDriversPerTask;
    in >> outputBufferBytes;
    in >> exchangeBufferBytes;
    in >> batchBytes;
    int32_t numVectors;
    in >> numVectors;

    char newLine;
    do {
      in.read(&newLine, 1);
    } while (newLine != 10);

    std::vector<RowVectorPtr> vectors;
    for (auto i = 0; i < numVectors; ++i) {
      auto vector = restoreVector(in, pool_.get());
      vectors.push_back(std::dynamic_pointer_cast<RowVector>(vector));
    }
    return runOne(
        vectors,
        numSourceTasks,
        numDestinationTasks,
        numDriversPerTask,
        outputBufferBytes,
        exchangeBufferBytes,
        batchBytes);
  }

  void seed(size_t seed) {
    currentSeed_ = seed;
    fuzzer_.reSeed(seed);
    rng_.seed(currentSeed_);
  }

 private:
  void saveRepro(
      std::vector<RowVectorPtr>& vectors,
      int32_t numSourceTasks,
      int32_t numDestinationTasks,
      int32_t numDriversPerTask,
      int32_t outputBufferBytes,
      int32_t exchangeBufferBytes,
      int32_t batchBytes) {
    auto filePath =
        fmt::format("{}/exchange_fuzzer_repro.{}", FLAGS_repro_path, getpid());
    std::ofstream out(filePath, std::ofstream::binary);
    out << numSourceTasks << " " << numDestinationTasks << " "
        << numDriversPerTask << " " << outputBufferBytes << " "
        << exchangeBufferBytes << " " << batchBytes << " " << vectors.size()
        << std::endl;
    for (auto& vector : vectors) {
      saveVector(*vector, out);
    }
    LOG(INFO)
        << "Saved repro. To replay, do velox_exchange_fuzzer_test --replay "
        << filePath;
  }

  static constexpr int64_t kMaxMemory = 6UL << 30; // 6GB

  int64_t randInt(int64_t min, int64_t max) {
    return boost::random::uniform_int_distribution<int64_t>(min, max)(rng_);
  }

  RowVectorPtr expectedChecksums(
      std::vector<RowVectorPtr> vectors,
      const std::vector<std::vector<TypePtr>>& rawInputTypes,
      int32_t width) {
    auto& rowType = vectors.front()->type()->as<TypeKind::ROW>();
    auto plan = exec::test::PlanBuilder()
                    .values(vectors, true)
                    .partialAggregation({}, makeAggregates(rowType, 1))
                    .localPartition({})
                    .finalAggregation()
                    .planNode();
    return exec::test::AssertQueryBuilder(plan).maxDrivers(width).copyResults(
        pool_.get());
  }

  std::vector<std::string> makeAggregates(const RowType& row, int firstColumn) {
    std::vector<std::string> aggregates;
    for (auto i = firstColumn; i < row.size(); ++i) {
      aggregates.push_back(fmt::format("checksum({})", row.nameOf(i)));
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

  struct VectorFuzzer::Options options_;
  VectorFuzzer fuzzer_;
  FuzzerGenerator rng_;
  size_t currentSeed_{0};

  std::unordered_map<std::string, std::string> configSettings_;
  // Serial number to differentiate consecutive benchmark repeats.
  static int32_t iteration_;
};

int32_t ExchangeFuzzer::iteration_;

int main(int argc, char** argv) {
  folly::init(&argc, &argv);
  memory::MmapAllocator::Options options;
  options.capacity = 20UL << 30;
  options.useMmapArena = true;
  options.mmapArenaCapacityRatio = 1;

  auto allocator = std::make_shared<memory::MmapAllocator>(options);
  memory::MemoryAllocator::setDefaultInstance(allocator.get());
  memory::MemoryManager::getInstance(memory::MemoryManagerOptions{
      .capacity = static_cast<int64_t>(options.capacity),
      .allocator = allocator.get()});

  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  serializer::presto::PrestoVectorSerde::registerVectorSerde();
  exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);

  ExchangeFuzzer fuzzer;

  if (!FLAGS_replay.empty()) {
    return fuzzer.replay() ? 0 : 1;
  }
  if (FLAGS_seed != 0) {
    LOG(INFO) << "Initial seed = " << FLAGS_seed;
    fuzzer.seed(FLAGS_seed);
  } else {
    size_t seed = getCurrentTimeMicro();
    LOG(INFO) << "Starting seed = " << seed;
    fuzzer.seed(seed);
  }
  fuzzer.run();
  return 0;
}
