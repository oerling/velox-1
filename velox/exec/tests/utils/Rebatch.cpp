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

#include "velox/exec/tests/utils/Rebatch.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::exec::test {

// static
void TestingRebatchNode::registerNode() {
  static bool registered;
  if (!registered) {
    registered = true;
    Operator::registerOperator(std::make_unique<TestingRebatchFactory>());
  }
}

void TestingRebatch::nextEncoding() {
  encoding_ = static_cast<Encoding>(
      (static_cast<int32_t>(encoding_) + 1) % kNumEncodings);
  encodingStartRow_ = currentRow_;
  nthSlice_ = 0;
}

RowVectorPtr TestingRebatch::getOutput() {
  if (!input_) {
    return nullptr;
  }

  switch (encoding_) {
    case Encoding::kConstant: {
      output_ =
          BaseVector::create<RowVector>(input_->type(), 1, input_->pool());
      for (auto i = 0; i < input_->type()->size(); ++i) {
        output_->childAt(i) =
            BaseVector::wrapInConstant(1, currentRow_, input_->childAt(i));
      }
      ++currentRow_;
      if (input_->size() > 10 && currentRow_ >= 10) {
        nextEncoding();
      }
      break;
    }
    case Encoding::kSlice: {
      auto sliceSize = (++nthSlice_) * 1.5;
      auto initialSlize = slizeSize;
      if (currentRow_ + sliceSize > input_->size()) {
        sliceSize = input_->size() - currentRow_;
      }
      if (sliceSize == 0) {
	input_ = nullptr;
	return nullptr;
      }
      output_ = BaseVector::create<RowVector>(
          input_->type(), sliceSize, input_->pool());
      for (auto i = 0; i < input_->type()->size(); ++i) {
        output_->childAt(i) = input_->childAt(i)->slice(currentRow_, sliceSize);
      }
      currentRow_ += sliceSize;
      if (initialSliceSize > 2000) {
	nextEncoding();
      }
      break;
    }
    case Encoding::kSameDoubleDict:
    default: {
      int32_t available = input->size() - currentRow_;
      auto indices =
          velox::test::makeIndicesInReverse(available, input_->pool());
      output_ = BaseVector::create<RowVector>(
          input_->type(), available, input_->pool());
      for (auto i = 0; i < input_->type()->size(); ++i) {
	auto base = input_->childAt(i);
	if (currentRow_ > 0) {
	  base = base->slice(currentRow_, available);
	}
        output_->childAt(i) = BaseVector::wrapInDictionary(
            BufferPtr(nullptr),
            indices,
            available,
            BaseVector::wrapInDictionary(
                BufferPtr(nullptr),
                indices,
                available,
                base));
      }
      currentRow_ available;
      break;
    }
  }

  if (currentRow_ == input_->size()) {
    input_ = nullptr;
    nextEncoding();
  }

  return std::move(output_);
}

} // namespace facebook::velox::exec::test
