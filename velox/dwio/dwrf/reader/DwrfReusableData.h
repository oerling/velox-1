

#include "velox/dwio/common/CacheInputStream.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/dwrf/common/ByteRLE.h"

#include <folly/container/F14Map.h>

namespace facebook::velox::dwrf {

class DwrfReusableData : public dwio::common::ScanReusableData {
 public:
  DwrfReusableData(
      const std::string id,
      memory::MemoryPool* pool,
      std::function<void(dwio::common::ScanReusableData*)> freeFunc)
      : ScanReusableData(id, pool, freeFunc) {}

  template <typename T>
  std::unique_ptr<T> getReader(TypeKind kind) {
    return nullptr;
  }

  template <typename T>
  void releasereader(std::unique_ptr<T>) {
    static_assert(std::is_base_of_v<dwio::common::SelectiveColumnReader, T>);
  }

 private:
  std::vector<std::unique_ptr<dwio::common::SeekableInputStream>> streams;
};

} // namespace facebook::velox::dwrf
