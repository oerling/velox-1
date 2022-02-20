


class SelectiveTimestampColumnReader : public SelectiveColumnReader {
 public:
  // The readers produce int64_t, the vector is Timestamps.
  using ValueType = int64_t;

  SelectiveTimestampColumnReader(
      const std::shared_ptr<const TypeWithId>& nodeType,
      StripeStreams& stripe,
      common::ScanSpec* scanSpec,
      FlatMapContext flaatMapContext);

  void seekToRowGroup(uint32_t index) override;
  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

 private:
  template <bool dense>
  void readHelper(RowSet rows);

  std::unique_ptr<IntDecoder</*isSigned*/ true>> seconds_;
  std::unique_ptr<IntDecoder</*isSigned*/ false>> nano_;

  // Values from copied from 'seconds_'. Nanos are in 'values_'.
  BufferPtr secondsValues_;
};

S
