

#include "velox/dwio/dwrf/reader/StructColumnReader.h"

namespace facebook::velox::dwrf {

lass ColumnLoader : public velox::VectorLoader {
 public:
  ColumnLoader(
      SelectiveStructColumnReader * structReader,
      SelectiveColumnReader * fieldReader,
      uint64_t version)
      : structReader_(structReader),
        fieldReader_(fieldReader),
        version_(version) {}

 protected:
  void loadInternal(RowSet rows, ValueHook * hook, VectorPtr * result) override;

 private:
  SelectiveStructColumnReader* structReader_;
  SelectiveColumnReader* fieldReader_;
  // This is checked against the version of 'structReader' on load. If
  // these differ, 'structReader' has been advanced since the creation
  // of 'this' and 'this' is no longer loadable.
  const uint64_t version_;
};

} // namespace facebook::velox::dwrf
