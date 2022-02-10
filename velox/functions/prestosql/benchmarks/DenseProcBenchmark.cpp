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

VELOX_UDF_BEGIN(scale_and_clamp)
FOLLY_ALWAYS_INLINE bool call(float& out, float f) {
  out = std::clamp(f + 17.0, -10.0, 10.0);
  return true;
}
VELOX_UDF_END();

  VectorPtr fused(const VectorPtr& data, FlatVectorPtr<float> result) {
  const auto numRows = data->size();
  auto rawResults = result->mutableRawValues<float>();

  auto features = data->as<RowVector>()->childAt(0)->asFlatVector<float>();
  auto rawFeatures = features->rawValues();

  for (auto row = 0; row < numRows; ++row) {
    rawResults[row] = std::clamp(rawFeatures[row] + 17.0, -10.0, 10.0);
  }

  return result;
}

    
  VectorPtr fusedAvx(const VectorPtr& data, FlatVectorPtr<float> result) {
  const auto numRows = data->size();
  auto rawResults = result->mutableRawValues<float>();

  auto features = data->as<RowVector>()->childAt(0)->asFlatVector<float>();
  auto rawFeatures = features->rawValues();
  using TV = simd::Vectors<float>;
  
  __m256 low = TV::setAll(-10);
  __m256 high = TV::setAll(10);
  __m256si allSet = simd::Vectors<int32_t>::setAll(-1);
  for (auto row = 0; row < numRows; row += 8) {
    __m256 values = *(__m256_u*)(rawFeatures + row) + 17.0;
    auto lowMask = TV::compareGt(low, values);
    auto highMask = TV::compareGt(values, high);
#if 1
    auto notHighMask = highMask ^ allSet;
    values = (__m256)(((__m256si)values & notHighMask ) | ((__m256si)high & highMask));
    auto notLowMask = lowMask ^ allSet;

		      *(__m256si_u*)(rawResults + row) = ((__m256si)values & notLowMask) | ((__m256si)low & lowMask);
#else
    float* resultPtr = rawResults + row;
    *(__m256_u*)resultPtr = values;
    _mm256_maskstore_ps(resultPtr, (__m256i)lowMask, low);
    _mm256_maskstore_ps(resultPtr, (__m256i)highMask, high);
#endif
  }

  return result;
}

  
class DenseProcBenchmark : public functions::test::FunctionBenchmarkBase {
 public:
  DenseProcBenchmark() : FunctionBenchmarkBase() {
    functions::prestosql::registerAllFunctions();
    registerFunction<udf_scale_and_clamp, float, float>({});
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

  void runVelox(const std::string& expression) {
    folly::BenchmarkSuspender suspender;
    auto input = vectorMaker_.rowVector({makeData()});
    auto exprSet = compileExpression(expression, input->type());
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
    int32_t numRows = input->size();
    auto result = std::static_pointer_cast<FlatVector<float>>(
      BaseVector::create(REAL(), numRows, input->pool()));

    suspender.dismiss();

    int cnt = 0;
    for (auto i = 0; i < 1'000; i++) {
      cnt += fused(input, result)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }

  void runFusedAvx() {
    folly::BenchmarkSuspender suspender;
    auto input = makeData();
    int32_t numRows = input->size();
    auto result = std::static_pointer_cast<FlatVector<float>>(
      BaseVector::create(REAL(), numRows, input->pool()));

    suspender.dismiss();

    volatile int cnt = 0;
    for (auto i = 0; i < 1'000; i++) {
      cnt += fusedAvx(input, result)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }

};

BENCHMARK(multipleFunctions) {
  DenseProcBenchmark benchmark;
  benchmark.runVelox(
      "clamp(c0.c0 + cast(17.0 as real), cast(-10 as real), cast(10 as real))");
}

BENCHMARK_RELATIVE(singleFunction) {
  DenseProcBenchmark benchmark;
  benchmark.runVelox("scale_and_clamp(c0.c0)");
}

BENCHMARK_RELATIVE(fusedExpressions) {
  DenseProcBenchmark benchmark;
  benchmark.runFused();
}

BENCHMARK_RELATIVE(fusedAvx) {
  DenseProcBenchmark benchmark;
  benchmark.runFusedAvx();
}

  
} // namespace

int main(int /*argc*/, char** /*argv*/) {
  folly::runBenchmarks();
  
    return 0;
}
