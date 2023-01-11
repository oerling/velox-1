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

#include <folly/Range.h>

namespace facebook::verax {

template <typename T, typename U>
bool isSubset(folly::Range<T*> subset, folly::Range<U*> superset) {
  for (auto item : subset) {
    if (std::find(
            superset.begin(), superset.end(), reinterpret_cast<U>(item)) ==
        superset.end()) {
      return false;
    }
  }
  return true;
}

// Returns how many leading members of 'ordered' are covered by 'set'
template <typename T, typename U>
int32_t prefixSize(folly::Range<T*> ordered, folly::Range<U*> set) {
  for (auto i = 0; i < ordered.size(); ++i) {
    if (std::find(set.begin(), set.end(), reinterpret_cast<U>(ordered[i])) ==
        set.end()) {
      return i;
    }
  }
  return ordered.size();
}

// Replaces each element of 'set' that matches an element of 'originals' with
// the corresponding element of 'replaceWith'.
template <typename T>
void replace(
    folly::Range<T*> set,
    folly::Range<T*> originals,
    T const* replaceWith) {
  for (auto& element : set) {
    auto it = std::find(originals.begin(), originals.end(), element);
    if (it == originals.end()) {
      continue;
    }
    element = replaceWith[it - originals.begin()];
  }
}

template <typename T, typename U>
void appendToVector(T& destination, U& source) {
  for (auto i : source) {
    destination.push_back(i);
  }
}

/// Returns index of 'expr' in collection 'exprs'. -1 if not found.
template <typename V>
int32_t position(const V& exprs, const Expr& expr) {
  for (auto i = 0; i < exprs.size(); ++i) {
    if (exprs[i]->sameOrEqual(expr)) {
      return i;
    }
  }
  return -1;
}

template <typename V, typename Getter>
int32_t position(const V& exprs, Getter getter, const Expr& expr) {
  for (auto i = 0; i < exprs.size(); ++i) {
    if (getter(exprs[i])->sameOrEqual(expr)) {
      return i;
    }
  }
  return -1;
}

/// Prints a number with precision' digits followed by a scale letter (n, u, m,
/// , k, M, G T, P.
std::string succinctNumber(double value, int32_t precision = 2);

std::string costString(float fanout, float cost, float setupCost);

} // namespace facebook::verax
