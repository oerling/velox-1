


class HashTableHolder : public exec::BaseHashTable {
 public:

  HashStringAllocator* stringAllocator() override { VELOX_UNREACHABLE(); }

  virtual void prepareForGroupProbe(
      HashLookup& lookup,
      const RowVectorPtr& input,
      SelectivityVector& rows,
      int8_t spillInputStartPartitionBit) { VELOX_UNREACHABLE(); }

  virtual void groupProbe(
      HashLookup& lookup,
      int8_t spillInputStartPartitionBit) override { VELOX_UNREACHABLE(); }

  virtual void joinProbe(HashLookup& lookup) override { VELOX_UNREACHABLE(); }

  virtual void prepareForJoinProbe(
      HashLookup& lookup,
      const RowVectorPtr& input,
      SelectivityVector& rows,
      bool decodeAndRemoveNulls) override { VELOX_UNREACHABLE(); }

  virtual int32_t listJoinResults(
      JoinResultIterator& iter,
      bool includeMisses,
      folly::Range<vector_size_t*> inputRows,
      folly::Range<char**> hits,
      uint64_t maxBytes) override { VELOX_UNREACHABLE(); }
  virtual int32_t listNotProbedRows(
      RowsIterator* iter,
      int32_t maxRows,
      uint64_t maxBytes,
      char** rows) override { VELOX_UNREACHABLE(); }
  virtual int32_t listProbedRows(
      RowsIterator* iter,
      int32_t maxRows,
      uint64_t maxBytes,
      char** rows) override { VELOX_UNREACHABLE(); }

  virtual int32_t listAllRows(
      RowsIterator* iter,
      int32_t maxRows,
      uint64_t maxBytes,
      char** rows) override { VELOX_UNREACHABLE(); }
  virtual int32_t listNullKeyRows(
      NullKeyRowsIterator* iter,
      int32_t maxRows,
      char** rows,
      const std::vector<std::unique_ptr<VectorHasher>>& hashers) override { VELOX_UNREACHABLE(); }

  virtual void prepareJoinTable(
      std::vector<std::unique_ptr<BaseHashTable>> tables,
      int8_t spillInputStartPartitionBit,
      folly::Executor* executor = nullptr) override { VELOX_UNREACHABLE(); }

  virtual int64_t allocatedBytes()  const override { VELOX_UNREACHABLE(); }

  virtual void clear(bool freeTable) override { VELOX_UNREACHABLE(); }
  virtual uint64_t capacity()  const override { VELOX_UNREACHABLE(); }

  virtual uint64_t numDistinct()  const override { VELOX_UNREACHABLE(); }
  virtual HashTableStats stats()  const override { VELOX_UNREACHABLE(); }
  virtual uint64_t hashTableSizeIncrease(int32_t numNewDistinct)  const override { VELOX_UNREACHABLE(); }
  virtual uint64_t estimateHashTableSize(uint64_t numDistinct)  const override { VELOX_UNREACHABLE(); }

  virtual bool hasDuplicateKeys()  const override { VELOX_UNREACHABLE(); }
  virtual HashMode hashMode() const { VELOX_UNREACHABLE();}


  virtual void decideHashMode(
      int32_t numNew,
      int8_t spillInputStartPartitionBit,
      bool disableRangeArrayHash = false) override { VELOX_UNREACHABLE(); }

  virtual void erase(folly::Range<char**> rows) override { VELOX_UNREACHABLE(); }
  virtual std::string toString() override { VELOX_UNREACHABLE(); }


  std::vector<RowContainer*> allRows() const override {VELOX_UNREACHABLE();}





  virtual void extractColumn(
      folly::Range<char* const*> rows,
      int32_t columnIndex,
      const VectorPtr& result) override { VELOX_UNREACHABLE(); }

 protected:

  virtual void setHashMode(
      HashMode mode,
      int32_t numNew,
      int8_t spillInputStartPartitionBit) override { VELOX_UNREACHABLE(); }

  std::vector<std::unique_ptr<VectorHasher>> hashers_;
  std::unique_ptr<RowContainer> rows_;

  // Time spent in build outside of the calling thread.
  CpuWallTiming offThreadBuildTiming_;

  std::shared_ptr<OperatorState> state();

 private:
  std::shared_ptr<OperatorState> state_;
  

};
