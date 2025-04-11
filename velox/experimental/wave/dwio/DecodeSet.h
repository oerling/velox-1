

using InterpretReturnFunc = std::function<void(DecodeSet* set, int32_t opIdx, int32_t level, DecodeStep* step)>;

using WidenFunc= std::function<void(DecodeSet* set, int32_t opIdx, int32_t level, int32_t numBlocks, DecodeStep* step,
std::vector<std::unique_ptr<DecodeStep>>& moreSteps)>;

  class DecodeSet {
 public:

    float levelCost(int32_t levelIdx);
    int32_t maxLevelBlocks(int32_t level);
    
  SplitStaging* splitStaging(int32_t level);
  ResultStaging* resultStaging(int32_t level);
  
  /// Adds an action into a decode plan.
  void addAction(
	    std::unique_ptr<Decodestep> step,
	    int32_t opIdx,
	    int32_t level,
	    DecodeStep* predecessor,
	    InterpretReturnFunc interpretFunc,
	    WidenFunc widen);

  /// Prepares a batch of decode. Applies to columns of 'ops' for the
  /// rows that are active in 'readStream'.  There can be selection
  /// from filters and joins on readStream in addition to selection
  /// from pushdown filters. If wrap is given, this is a Operand that contains the wrap that applies to columns assigned in table scan. 
  SyncFlag getActions(bool filters, operand* wrap, std::vector<ColumnOp*> ops, ReadStream& readStream, DecodePrograms& result, std::vector<SplitStaging*> toLoad);

  std::vector<std::unique_ptr<DecodeStep> allSteps_;
  std::vector<std::vector<DecodeStep*>> stepByOp_;
  std::vector<std::unique_ptr<SplitStaging> splitStaging_;
  
  };
  

