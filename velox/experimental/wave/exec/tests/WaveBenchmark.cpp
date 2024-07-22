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


#include "velox/benchmark/QueryBenchmarkBase.h"
#include "velox/experimental/wave/exec/tests/utils/FileFormat.h"



using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;


DEFINE_string(
    data_path,
    "",
    "Root path of test data. Data layout must follow Hive-style partitioning. "
    "If the content are directories, they contain the data files for "
    "each table. If they are files, they contain a file system path for each "
    "data file, one per line. This allows running against cloud storage or "
    "HDFS");
namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}
} // namespace

DEFINE_validator(data_path, &notEmpty);

DEFINE_bool(wave, true, "Run benchmark with Wave");

DEFINE_int32(
    run_query_verbose,
    -1,
    "Run a given query and print execution statistics");


class WaveBenchmark : public QueryBenchmarkBase {
 public:
  void initialize() override {
    if (FLAGS_wave) {
      if (int device; cudaGetDevice(&device) != cudaSuccess) {
	GTEST_SKIP() << "No CUDA detected, skipping all tests";
      }
      wave::registerWave();
      wave::WaveHiveDataSource::registerConnector();
      wave::test::WaveTestSplitReader::registerTestSplitReader();
    }
  }

  auto makeData(
      const RowTypePtr& type,
      int32_t numVectors,
      int32_t vectorSize,
      bool notNull = true) {
    vectors_ = makeVectors(type, numVectors, vectorSize);
    int32_t cnt = 0;
    for (auto& vector : vectors_) {
      makeRange(vector, 1000000000, notNull);
      auto rn = vector->childAt(type->size() - 1)->as<FlatVector<int64_t>>();
      for (auto i = 0; i < rn->size(); ++i) {
        rn->set(i, cnt++);
      }
    }
    auto splits = makeTable("test", vectors_);
    return splits;
  }

  wave::test::SplitVector makeTable(
      const std::string& name,
      std::vector<RowVectorPtr>& rows) {
    wave::test::Table::dropTable(name);
    return wave::test::Table::defineTable(name, rows)->splits();
  }

void writeToFile(
    const std::string& filePath,
    const std::vector<RowVectorPtr>& vectors,
    std::shared_ptr<dwrf::Config> config,
    const TypePtr& schema) {
  velox::dwrf::WriterOptions options;
  options.config = config;
  options.schema = schema;
  auto localWriteFile = std::make_unique<LocalWriteFile>(filePath, true, false);
  auto sink = std::make_unique<dwio::common::WriteFileSink>(
      std::move(localWriteFile), filePath);
  auto childPool = rootPool_->addAggregateChild("HiveConnectorTestBase.Writer");
  options.memoryPool = childPool.get();
  facebook::velox::dwrf::Writer writer{std::move(sink), options};
  for (size_t i = 0; i < vectors.size(); ++i) {
    writer.write(vectors[i]);
  }
  writer.close();
}

  TpChPlan getQueryPlan(int32_t query) {
    
  }

  std::vector<std::shared_ptr><connector::ConnectorSplit>> listSplits(const std::string& path, int32_t numSplitsPerFile, const TpchPlan& plan) override {
    if (plan.fileFormat == FileFormat::UNKNOWN){
      auto table = wave::test::Table::getTable(path);
      return table.splits();
    }
    return QueryBenchmarkBase::listSplits(path, numSplitsPerFile, plan);
}


  
  void runMain(std::ostream& out, RunStats& runStats) override {
    if (FLAGS_run_query_verbose == -1) {
      folly::runBenchmarks();
    } else {
      const auto queryPlan = 
         getQueryPlan(FLAGS_run_query_verbose);
      auto [cursor, actualResults] = run(queryPlan);
      if (!cursor) {
        LOG(ERROR) << "Query terminated with error. Exiting";
        exit(1);
      }
      auto task = cursor->task();
      ensureTaskCompletion(task.get());
      if (FLAGS_include_results) {
        printResults(actualResults, out);
        out << std::endl;
      }
      const auto stats = task->taskStats();
      int64_t rawInputBytes = 0;
      for (auto& pipeline : stats.pipelineStats) {
        auto& first = pipeline.operatorStats[0];
        if (first.operatorType == "TableScan") {
          rawInputBytes += first.rawInputBytes;
        }
      }
      runStats.rawInputBytes = rawInputBytes;
      out << fmt::format(
                 "Execution time: {}",
                 succinctMillis(
                     stats.executionEndTimeMs - stats.executionStartTimeMs))
          << std::endl;
      out << fmt::format(
                 "Splits total: {}, finished: {}",
                 stats.numTotalSplits,
                 stats.numFinishedSplits)
          << std::endl;
      out << printPlanWithStats(
                 *queryPlan.plan, stats, FLAGS_include_custom_stats)
          << std::endl;
    }
  }
};

WaveBenchmark benchmark;

BENCHMARK(q1) {
  const auto planContext = queryBuilder->getQueryPlan(1);
  benchmark.run(planContext);
}

BENCHMARK(q2) {
  const auto planContext = queryBuilder->getQueryPlan(2);
  benchmark.run(planContext);
}

BENCHMARK(q3) {
  const auto planContext = queryBuilder->getQueryPlan(3);
  benchmark.run(planContext);
}

BENCHMARK(q5) {
  const auto planContext = queryBuilder->getQueryPlan(5);
  benchmark.run(planContext);
}

BENCHMARK(q6) {
  const auto planContext = queryBuilder->getQueryPlan(6);
  benchmark.run(planContext);
}

BENCHMARK(q7) {
  const auto planContext = queryBuilder->getQueryPlan(7);
  benchmark.run(planContext);
}

BENCHMARK(q8) {
  const auto planContext = queryBuilder->getQueryPlan(8);
  benchmark.run(planContext);
}

BENCHMARK(q9) {
  const auto planContext = queryBuilder->getQueryPlan(9);
  benchmark.run(planContext);
}

BENCHMARK(q10) {
  const auto planContext = queryBuilder->getQueryPlan(10);
  benchmark.run(planContext);
}

BENCHMARK(q11) {
  const auto planContext = queryBuilder->getQueryPlan(11);
  benchmark.run(planContext);
}

BENCHMARK(q12) {
  const auto planContext = queryBuilder->getQueryPlan(12);
  benchmark.run(planContext);
}

BENCHMARK(q13) {
  const auto planContext = queryBuilder->getQueryPlan(13);
  benchmark.run(planContext);
}

BENCHMARK(q14) {
  const auto planContext = queryBuilder->getQueryPlan(14);
  benchmark.run(planContext);
}

BENCHMARK(q15) {
  const auto planContext = queryBuilder->getQueryPlan(15);
  benchmark.run(planContext);
}

BENCHMARK(q16) {
  const auto planContext = queryBuilder->getQueryPlan(16);
  benchmark.run(planContext);
}

BENCHMARK(q17) {
  const auto planContext = queryBuilder->getQueryPlan(17);
  benchmark.run(planContext);
}

BENCHMARK(q18) {
  const auto planContext = queryBuilder->getQueryPlan(18);
  benchmark.run(planContext);
}

BENCHMARK(q19) {
  const auto planContext = queryBuilder->getQueryPlan(19);
  benchmark.run(planContext);
}

BENCHMARK(q20) {
  const auto planContext = queryBuilder->getQueryPlan(20);
  benchmark.run(planContext);
}

BENCHMARK(q21) {
  const auto planContext = queryBuilder->getQueryPlan(21);
  benchmark.run(planContext);
}

BENCHMARK(q22) {
  const auto planContext = queryBuilder->getQueryPlan(22);
  benchmark.run(planContext);
}

int WaveBenchmarkMain() {
  benchmark.initialize();
  queryBuilder =
      std::make_shared<TpchQueryBuilder>(toFileFormat(FLAGS_data_format));
  queryBuilder->initialize(FLAGS_data_path);
  if (FLAGS_test_flags_file.empty()) {
    RunStats ignore;
    benchmark.runMain(std::cout, ignore);
  } else {
    benchmark.runAllCombinations();
  }
  benchmark.shutdown();
  queryBuilder.reset();
  return 0;
}

