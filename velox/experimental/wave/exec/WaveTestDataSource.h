

namespace facebook::velox::wave {

/// A WaveDataSource that decodes mock Wave tables.
class WaveTestDataSource : public WaveDataSource {
 public:
 WaveTestDataSource(const std::shared_ptr<WaveTestConnectorSplit>& split)
    : split_(split);
  
 private:
  std::shared_pptr<WaveTestConnectorSplit> split_;
};
}

