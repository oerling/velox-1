

#include "velox/dwio/dwrf/reader/SelectiveStructColumnReader.h"

namespace facebook::velox::dwrf {
// Wraps '*result' in a dictionary to make the contiguous values
// appear at the indices i 'rows'. Used when loading a LazyVector for
// a sparse set of rows in conditional exprs.
static void scatter(RowSet rows, VectorPtr* result) {
  auto end = rows.back() + 1;
  // Initialize the indices to 0 to make the dictionary safely
  // readable also for uninitialized positions.
  auto indices =
      AlignedBuffer::allocate<vector_size_t>(end, (*result)->pool(), 0);
  auto rawIndices = indices->asMutable<vector_size_t>();
  for (int32_t i = 0; i < rows.size(); ++i) {
    rawIndices[rows[i]] = i;
  }
  *result =
      BaseVector::wrapInDictionary(BufferPtr(nullptr), indices, end, *result);
}

void ColumnLoader::loadInternal(
    RowSet rows,
    ValueHook* hook,
    VectorPtr* result) {
  VELOX_CHECK_EQ(
      version_,
      structReader_->numReads(),
      "Loading LazyVector after the enclosing reader has moved");
  auto offset = structReader_->lazyVectorReadOffset();
  auto incomingNulls = structReader_->nulls();
  auto outputRows = structReader_->outputRows();
  raw_vector<vector_size_t> selectedRows;
  RowSet effectiveRows;
  if (rows.size() == outputRows.size()) {
    // All the rows planned at creation are accessed.
    effectiveRows = outputRows;
  } else {
    // rows is a set of indices into outputRows. There has been a
    // selection between creation and loading.
    selectedRows.resize(rows.size());
    assert(!selectedRows.empty());
    for (auto i = 0; i < rows.size(); ++i) {
      selectedRows[i] = outputRows[rows[i]];
    }
    effectiveRows = RowSet(selectedRows);
  }

  structReader_->advanceFieldReader(fieldReader_, offset);
  fieldReader_->scanSpec()->setValueHook(hook);
  fieldReader_->read(offset, effectiveRows, incomingNulls);
  if (!hook) {
    fieldReader_->getValues(effectiveRows, result);
    if (rows.size() != outputRows.size()) {
      // We read sparsely. The values that were read should appear
      // at the indices in the result vector that were given by
      // 'rows'.
      scatter(rows, result);
    }
  }
}

}
