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

#pragma once

#include <cstdint>

namespace facebook::velox {
#define MTRT(t) MTRACK(sizeof(t))
#define MTRN(t, n) MTRACK((t) * (n))
#define MTRACK(n) facebook::velox::mtrack(__FILE__, __LINE__, n)
#define TPSH(v)                                                    \
  {                                                                \
    if (v.capacity() <= v.size() + 1) {                            \
      mtrack(__FILE__, __LINE__, sizeof(v[0]) * v.capacity() * 2); \
    }                                                              \
  }

void mtrack(const char* file, int line, long bytes);

} // namespace facebook::velox
