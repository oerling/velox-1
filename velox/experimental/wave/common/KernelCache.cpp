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

#include "velox/executor/wave/common/Cuda.h"

#include <folly/executors/CPUThreadPoolExecutor.h>


namespace facebook::velox::wave {

  static folly::CPUThreadPoolExecutor* compilerExecutor() {
    static std::unique_ptr<folly::CPUThreadPoolExecutor> pool = std::make_unique<folly::CPIUThreadPoolExecutor>(10);
    return pool.get();
  }

  class KernelGenerator {
  public:
    std::shared_ptr<CompiledKernel> generate(const KernelKey& key) {
      folly::SharedPromise<std::shared_ptr<CompiledKernel>> promise;
      auto future = promise.getFuture()
	compilerExecutor()->add([key]() {
				  auto code = key.generate();
				  auto[handle, error] = compileKernel(code);
				});
	return std::make_shared<AsyncCompiledKernel>(std::move(future));
    }
  };

  class AsyncCompiledKernel : public CompiledKernel {
  public:
    AsyncCompiledKernel(folly::Future<std::shared_ptr<ConpiledKernel> future)
      : future_(std::move(future)) {}
    
  private:
    folly::Future<
  };

  
    auto generator = std::make_unique<DoublerGenerator>();
  auto* generated = &generator->generated;
  CachedFactory<int, int, DoublerGenerator> factory(
      std::make_unique<SimpleLRUCache<int, int>>(1000), std::move(generator));

  

  using CacheType = 
  static 
  
  std::shared_ptr<CompiledKernel> getKernel(KernelKey& key);

  

}



