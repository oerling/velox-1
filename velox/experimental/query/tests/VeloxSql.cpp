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

#include <folly/init/Init.h>
#include <gflags/gflags.h>

#include "velox/common/base/SuccinctPrinter.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MmapAllocator.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/Split.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/experimental/query/LocalRunner.h"
#include "velox/experimental/query/LocalSchema.h"
#include "velox/experimental/query/Plan.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/QueryPlanner.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}

static bool validateDataFormat(const char* flagname, const std::string& value) {
  if ((value.compare("parquet") == 0) || (value.compare("dwrf") == 0)) {
    return true;
  }
  std::cout
      << fmt::format(
             "Invalid value for --{}: {}. Allowed values are [\"parquet\", \"dwrf\"]",
             flagname,
             value)
      << std::endl;
  return false;
}

void printResults(const std::vector<RowVectorPtr>& results) {
  std::cout << "Results:" << std::endl;
  bool printType = true;
  for (const auto& vector : results) {
    // Print RowType only once.
    if (printType) {
      std::cout << vector->type()->asRow().toString() << std::endl;
      printType = false;
    }
    for (vector_size_t i = 0; i < vector->size(); ++i) {
      std::cout << vector->toString(i) << std::endl;
    }
  }
}
} // namespace

DEFINE_string(
    data_path,
    "",
    "Root path of data. Data layout must follow Hive-style partitioning. ");

DEFINE_int32(optimizer_trace, 0, "Optimizer trace level");

DEFINE_bool(print_stats, false, "print statistics");
DEFINE_bool(
    include_custom_stats,
    false,
    "Include custom statistics along with execution statistics");
DEFINE_bool(include_results, false, "Include results in the output");
DEFINE_bool(use_native_parquet_reader, true, "Use Native Parquet Reader");
DEFINE_int32(num_drivers, 4, "Number of drivers");
DEFINE_int32(num_workers, 4, "Number of in-process workers");

DEFINE_string(data_format, "parquet", "Data format");
DEFINE_int32(num_splits_per_file, 10, "Number of splits per file");
DEFINE_int32(
    cache_gb,
    0,
    "GB of process memory for cache and query.. if "
    "non-0, uses mmap to allocator and in-process data cache.");
DEFINE_int32(num_repeats, 1, "Number of times to run --query");
DEFINE_string(
    query,
    "",
    "Text of query. If empty, reads ';' separated queries from standard input");

DEFINE_validator(data_path, &notEmpty);
DEFINE_validator(data_format, &validateDataFormat);

class VeloxRunner {
 public:
  void initialize() {
    if (FLAGS_cache_gb) {
      int64_t memoryBytes = FLAGS_cache_gb * (1LL << 30);
      memory::MmapAllocator::Options options;
      options.capacity = memoryBytes;
      options.useMmapArena = true;
      options.mmapArenaCapacityRatio = 1;

      auto allocator = std::make_shared<memory::MmapAllocator>(options);
      allocator_ = std::make_shared<cache::AsyncDataCache>(
          allocator, memoryBytes, nullptr);
      memory::MemoryAllocator::setDefaultInstance(allocator_.get());
    }
    pool_ = memory::getProcessDefaultMemoryManager().getPool()->addChild(
        "interactive_sql");
    functions::prestosql::registerAllScalarFunctions();
    aggregate::prestosql::registerAllAggregateFunctions();
    parse::registerTypeResolver();
    filesystems::registerLocalFileSystem();
    if (FLAGS_use_native_parquet_reader) {
      parquet::registerParquetReaderFactory(parquet::ParquetReaderType::NATIVE);
    } else {
      parquet::registerParquetReaderFactory(parquet::ParquetReaderType::DUCKDB);
    }
    dwrf::registerDwrfReaderFactory();
    ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(8);

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, nullptr, ioExecutor_.get());
    connector::registerConnector(hiveConnector);
    schema_ = std::make_unique<facebook::verax::LocalSchema>(
        FLAGS_data_path, toFileFormat(FLAGS_data_format), kHiveConnectorId);
    planner_ = std::make_unique<core::DuckDbQueryPlanner>(pool_.get());
    for (auto& pair : schema_->tables()) {
      planner_->registerTable(pair.first, pair.second->rowType());
    }
    planner_->registerTableScan([this](
                                    const std::string& id,
                                    const std::string& name,
                                    const RowTypePtr& rowType) {
      return toTableScan(id, name, rowType);
    });
    splitSourceFactory_ = std::make_unique<LocalSplitSourceFactory>(
        *schema_, FLAGS_num_splits_per_file);
  }

  core::PlanNodePtr toTableScan(
      const std::string& id,
      const std::string& name,
      const RowTypePtr& rowType) {
    using namespace connector::hive;
    auto handle = std::make_shared<HiveTableHandle>(
        kHiveConnectorId, name, true, SubfieldFilters{}, nullptr);
    std::unordered_map<std::string, std::shared_ptr<connector::ColumnHandle>>
        assignments;

    auto table = schema_->findTable(name);
    for (auto i = 0; i < rowType->size(); ++i) {
      auto column = rowType->nameOf(i);
      VELOX_CHECK(
          table->columns.find(column) != table->columns.end(),
          "No column {} in {}",
          column,
          name);
      assignments[column] = std::make_shared<HiveColumnHandle>(
          column, HiveColumnHandle::ColumnType::kRegular, rowType->childAt(i));
    }
    return std::make_shared<core::TableScanNode>(
        id, rowType, handle, assignments);
  }

  void run(const std::string& sql) {
    core::PlanNodePtr plan;
    try {
      plan = planner_->plan(sql);
    } catch (std::exception& e) {
      std::cout << "parse error: " << e.what();
      return;
    }
    std::vector<ExecutableFragment> fragments;
    ExecutablePlanOptions opts;
    opts.numWorkers = FLAGS_num_workers;
    opts.numDrivers = FLAGS_num_drivers;

    try {
      auto allocator = std::make_unique<HashStringAllocator>(pool_.get());
      auto context =
          std::make_unique<facebook::verax::QueryGraphContext>(*allocator);
      facebook::verax::queryCtx() = context.get();
      facebook::verax::Schema veraxSchema("test", schema_.get());
      facebook::verax::Optimization opt(
          *plan, veraxSchema, FLAGS_optimizer_trace);
      auto best = opt.bestPlan();
      fragments = opt.toVeloxPlan(best->op, opts);
      facebook::verax::queryCtx() = nullptr;
    } catch (const std::exception& e) {
      facebook::verax::queryCtx() = nullptr;
      std::cout << "Optimization error: " << e.what();
      return;
    }

    try {
      LocalRunner runner(
          std::move(fragments), queryCtx_, splitSourceFactory_.get(), opts);
      auto cursor = runner.cursor();
      std::vector<VectorPtr> result;
      while (cursor->moveNext()) {
        result.push_back(cursor->current());
      }
      waitForTaskCompletion(cursor->task().get());
    } catch (const std::exception& e) {
      LOG(ERROR) << "Query terminated with: " << e.what();
      return;
    }
  }

  std::shared_ptr<memory::MemoryAllocator> allocator_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;
  std::shared_ptr<core::QueryCtx> queryCtx_;
  std::unique_ptr<facebook::verax::LocalSchema> schema_;
  std::unique_ptr<LocalSplitSourceFactory> splitSourceFactory_;
  std::unique_ptr<core::DuckDbQueryPlanner> planner_;
};

int main(int argc, char** argv) {
  std::string kUsage(
      "Velox local SQL command line. Run 'velox_sql --help' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::init(&argc, &argv, false);
  VeloxRunner runner;
  runner.initialize();
  runner.run(FLAGS_query);
  return 0;
}
