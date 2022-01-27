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
#include "velox/expression/tests/VectorFuzzer.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

VectorPtr fused(const VectorPtr& data) {
  const auto numRows = data->size();
  auto result = std::static_pointer_cast<FlatVector<float>>(
      BaseVector::create(REAL(), numRows, data->pool()));
  auto rawResults = result->mutableRawValues<float>();

  auto features = data->as<RowVector>()->childAt(0)->asFlatVector<float>();
  auto rawFeatures = features->rawValues();

  for (auto row = 0; row < numRows; ++row) {
    rawResults[row] = std::clamp(rawFeatures[row] + 17.0, -10.0, 10.0);
  }

  return result;
}

class DenseProcBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  DenseProcBenchmark() : FunctionBenchmarkBase() {
    functions::prestosql::registerAllFunctions();
  }

  RowVectorPtr makeData() {
    VectorFuzzer::Options opts;
    opts.vectorSize = 256;
    return vectorMaker_.rowVector({
        VectorFuzzer(opts, pool()).fuzzFlat(REAL()),
        VectorFuzzer(opts, pool()).fuzzFlat(REAL()),
        VectorFuzzer(opts, pool()).fuzzFlat(REAL()),
    });
  }

  void runSeparate() {
    folly::BenchmarkSuspender suspender;
    auto input = vectorMaker_.rowVector({makeData()});
    auto exprSet = compileExpression(
        "clamp(c0.c0 + cast(17.0 as real), cast(-10 as real), cast(10 as real))",
        input->type());
    SelectivityVector rows(input->size());
    suspender.dismiss();

    doRun(rows, exprSet, input);
  }

  VectorPtr evaluate2(
      const SelectivityVector& rows,
      exec::ExprSet& exprSet,
      const RowVectorPtr& data) {
    exec::EvalCtx evalCtx(&execCtx_, &exprSet, data.get());
    std::vector<VectorPtr> results(1);
    exprSet.eval(rows, &evalCtx, &results);
    return results[0];
  }

  void doRun(
      const SelectivityVector& rows,
      ExprSet& exprSet,
      const RowVectorPtr& rowVector) {
    int cnt = 0;
    for (auto i = 0; i < 1'000; i++) {
      cnt += evaluate2(rows, exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }

  void runFused() {
    folly::BenchmarkSuspender suspender;
    auto input = makeData();
    suspender.dismiss();

    int cnt = 0;
    for (auto i = 0; i < 1'000; i++) {
      cnt += fused(input)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

BENCHMARK(separateExpressions) {
  DenseProcBenchmark benchmark;
  benchmark.runSeparate();
}

BENCHMARK_RELATIVE(fusedExpressions) {
  DenseProcBenchmark benchmark;
  benchmark.runFused();
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  return 0;
}
