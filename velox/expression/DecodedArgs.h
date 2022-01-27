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
#include "velox/expression/EvalCtx.h"

namespace facebook::velox::exec {

/// A helper class to decode VectorFunction arguments.
/// Example:
///    DecodedArgs decodedArgs(rows, args, context);
///    auto base = decodedArgs.at(0);
///    auto exp = decodedArgs.at(1);
///
///    rows.applyToSelected([&](int row) {
///      rawResults[row] =
///        std::pow(base->valueAt<double>(row), exp->valueAt<double>(row));
///    });
///
class DecodedArgs {
 public:
  DecodedArgs(
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      exec::EvalCtx* context) {
    flatVectors_.resize(args.size());
    for (auto i = 0; i < args.size(); ++i) {
      if (args[i]->encoding() == VectorEncoding::Simple::FLAT) {
        flatVectors_[i] = args[i].get();
      } else {
        flatVectors_[i] = nullptr;
      }
    }
    for (auto& arg : args) {
      if (arg->encoding() == VectorEncoding::Simple::FLAT) {
        holders_.emplace_back(context);
      } else {
        holders_.emplace_back(context, *arg.get(), rows);
      }
    }
  }

  BaseVector* flatAt(int i) const {
    return flatVectors_[i];
  }

  DecodedVector* at(int i) const {
    return const_cast<exec::LocalDecodedVector*>(&holders_[i])->get();
  }

  size_t size() const {
    return holders_.size();
  }

 private:
  std::vector<BaseVector*> flatVectors_;
  std::vector<exec::LocalDecodedVector> holders_;
};
} // namespace facebook::velox::exec
