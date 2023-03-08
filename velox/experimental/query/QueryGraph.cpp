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

#include "velox/experimental/query/QueryGraph.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/base/SuccinctPrinter.h"
#include "velox/experimental/query/Plan.h"
#include "velox/experimental/query/PlanUtils.h"

namespace facebook::verax {

size_t PlanObjectPtrHasher::operator()(const PlanObjectConstPtr& object) const {
  return object->hash();
}

bool PlanObjectPtrComparer::operator()(
    const PlanObjectConstPtr& lhs,
    const PlanObjectConstPtr& rhs) const {
  if (rhs == lhs) {
    return true;
  }
  return rhs && lhs && lhs->isExpr() && rhs->isExpr() &&
      reinterpret_cast<const Expr*>(lhs)->sameOrEqual(
          *reinterpret_cast<const Expr*>(rhs));
}

size_t PlanObject::hash() const {
  size_t h = static_cast<size_t>(id_);
  for (auto& child : children()) {
    h = velox::bits::hashMix(h, child->hash());
  }
  return h;
}

PlanObjectPtr QueryGraphContext::dedup(PlanObjectPtr object) {
  auto pair = deduppedObjects_.insert(object);
  return *pair.first;
}

const char* QueryGraphContext::toName(std::string_view str) {
  auto it = names_.find(str);
  if (it != names_.end()) {
    return it->data();
  }
  char* data = allocator_.allocate(str.size() + 1)->begin();
  memcpy(data, str.data(), str.size());
  data[str.size()] = 0;
  names_.insert(std::string_view(data, str.size()));
  return data;
}

const char* toName(const std::string& str) {
  return queryCtx()->toName(std::string_view(str.data(), str.size()));
}

float Value::byteSize() const {
  if (type->isFixedWidth()) {
    return type->cppSizeInBytes();
  }
  switch (type->kind()) {
      // Add complex types here.
    default:
      return 16;
  }
}
namespace {
template <typename V>
bool isZero(const V& bits, size_t begin, size_t end) {
  for (auto i = begin; i < end; ++i) {
    if (bits[i]) {
      return false;
    }
  }
  return true;
}
} // namespace

bool PlanObjectSet::operator==(const PlanObjectSet& other) const {
  // The sets are equal if they have the same bits set. Trailing words of zeros
  // do not count.
  auto l1 = bits_.size();
  auto l2 = other.bits_.size();
  for (auto i = 0; i < l1 && i < l2; ++i) {
    if (bits_[i] != other.bits_[i]) {
      return false;
    }
  }
  if (l1 < l2) {
    return isZero(other.bits_, l1, l2);
  }
  if (l2 < l1) {
    return isZero(bits_, l2, l1);
  }
  return true;
}

bool PlanObjectSet::isSubset(const PlanObjectSet& super) const {
  auto l1 = bits_.size();
  auto l2 = super.bits_.size();
  for (auto i = 0; i < l1 && i < l2; ++i) {
    if (bits_[i] & ~super.bits_[i]) {
      return false;
    }
  }
  if (l2 < l1) {
    return isZero(bits_, l2, l1);
  }
  return true;
}

size_t PlanObjectSet::hash() const {
  // The hash is a mix of the hashes of all non-zero words.
  size_t hash = 123;
  for (auto i = 0; i < bits_.size(); ++i) {
    hash = velox::simd::crc32U64(hash, bits_[i]);
  }
  return hash * hash;
}

void PlanObjectSet::unionColumns(ExprPtr expr) {
  switch (expr->type()) {
    case PlanType::kLiteral:
      return;
    case PlanType::kColumn:
      add(expr);
      return;
    case PlanType::kAggregate: {
      auto condition = expr->as<Aggregate>()->condition();
      if (condition) {
        unionColumns(condition);
      }
    }
      // Fall through.
    FOLLY_FALLTHROUGH; case PlanType::kCall: {
      auto call = reinterpret_cast<const Call*>(expr);
      unionSet(call->columns());
      return;
    }
    default:
      VELOX_UNREACHABLE();
  }
}

void PlanObjectSet::unionColumns(const ExprVector& exprs) {
  for (auto& expr : exprs) {
    unionColumns(expr);
  }
}

void PlanObjectSet::unionSet(const PlanObjectSet& other) {
  ensureWords(other.bits_.size());
  for (auto i = 0; i < other.bits_.size(); ++i) {
    bits_[i] |= other.bits_[i];
  }
}

void PlanObjectSet::intersect(const PlanObjectSet& other) {
  bits_.resize(std::min(bits_.size(), other.bits_.size()));
  for (auto i = 0; i < bits_.size(); ++i) {
    bits_[i] &= other.bits_[i];
  }
}

std::string PlanObjectSet::toString(bool names) const {
  std::stringstream out;
  forEach([&](auto object) {
    out << object->id();
    if (names) {
      out << ": " << object->toString() << std::endl;
    } else {
      out << " ";
    }
  });
  return out.str();
}

void Column::equals(ColumnPtr other) const {
  if (!equivalence_ && !other->equivalence_) {
    Declare(Equivalence, equiv);
    equiv->columns.push_back(this);
    equiv->columns.push_back(other);
    equivalence_ = equiv;
    other->equivalence_ = equiv;
    return;
  }
  if (!other->equivalence_) {
    other->equivalence_ = equivalence_;
    equivalence_->columns.push_back(other);
    return;
  }
  if (!equivalence_) {
    other->equals(this);
    return;
  }
  for (auto& column : other->equivalence_->columns) {
    equivalence_->columns.push_back(column);
    column->equivalence_ = equivalence_;
  }
}

std::string Column::toString() const {
  Name cname = !relation_ ? ""
      : relation_->type() == PlanType::kTable
      ? relation_->as<BaseTable>()->cname
      : relation_->type() == PlanType::kDerivedTable
      ? relation_->as<DerivedTable>()->cname
      : "--";

  return fmt::format("{}.{}", cname, name_);
}

std::string Call::toString() const {
  std::stringstream out;
  out << name_ << "(";
  for (auto i = 0; i < args_.size(); ++i) {
    out << args_[i]->toString() << (i == args_.size() - 1 ? ")" : ", ");
  }
  return out.str();
}

std::string conjunctsToString(const ExprVector& conjuncts) {
  std::stringstream out;
  for (auto i = 0; i < conjuncts.size(); ++i) {
    out << conjuncts[i]->toString()
        << (i == conjuncts.size() - 1 ? "" : " and ");
  }
  return out.str();
}

std::string BaseTable::toString() const {
  std::stringstream out;
  out << "{" << PlanObject::toString();
  out << schemaTable->name << " " << cname << "}";
  return out.str();
}

const JoinSide JoinEdge::sideOf(PlanObjectConstPtr side, bool other) const {
  if ((side == rightTable_ && !other) || (side == leftTable_ && other)) {
    return {
        rightTable_,
        rightKeys_,
        lrFanout_,
        rightOptional_,
        rightExists_,
        rightNotExists_,
        rightUnique_};
  }
  return {
      leftTable_,
      leftKeys_,
      rlFanout_,
      leftOptional_,
      false,
      false,
      leftUnique_};
}

void JoinEdge::addEquality(ExprPtr left, ExprPtr right) {
  leftKeys_.push_back(left);
  rightKeys_.push_back(right);
  guessFanout();
}

std::string JoinEdge::toString() const {
  std::stringstream out;
  out << "<join "
      << (leftTable_ ? leftTable_->toString() : " multiple tables ");
  if (leftOptional_ && rightOptional_) {
    out << " full outr ";
  } else if (rightExists_ && rightOptional_) {
    out << " exists project ";
  } else if (rightOptional_) {
    out << " exists ";
  } else if (rightOptional_) {
    out << " left outer ";
  } else if (rightNotExists_) {
    out << " not exists ";
  } else {
    out << " inner ";
  }
  out << rightTable_->toString();
  out << " on ";
  for (auto i = 0; i < leftKeys_.size(); ++i) {
    out << leftKeys_[i]->toString() << " = " << rightKeys_[i]->toString()
        << (i < leftKeys_.size() - 1 ? " and " : "");
  }
  if (!filter_.empty()) {
    out << " filter " << conjunctsToString(filter_);
  }
  out << ">";
  return out.str();
}

bool Expr::sameOrEqual(const Expr& other) const {
  if (this == &other) {
    return true;
  }
  if (type() != other.type()) {
    return false;
  }
  switch (type()) {
    case PlanType::kColumn:
      return as<Column>()->equivalence() &&
          as<Column>()->equivalence() == other.as<Column>()->equivalence();
    case PlanType::kAggregate: {
      auto a = reinterpret_cast<const Aggregate*>(this);
      auto b = reinterpret_cast<const Aggregate*>(&other);
      if (a->isDistinct() != b->isDistinct() ||
          a->isAccumulator() != b->isAccumulator() ||
          !(a->condition() == b->condition() ||
            (a->condition() && b->condition() &&
             a->condition()->sameOrEqual(*b->condition())))) {
        return false;
      }
    }
      // Fall through.
    FOLLY_FALLTHROUGH; case PlanType::kCall: {
      if (as<Call>()->name() != other.as<Call>()->name()) {
        return false;
      }
      auto numArgs = as<Call>()->args().size();
      if (numArgs != other.as<Call>()->args().size()) {
        return false;
      }
      for (auto i = 0; i < numArgs; ++i) {
        if (as<Call>()->args()[i]->sameOrEqual(*other.as<Call>()->args()[i])) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

PlanObjectConstPtr singleTable(PlanObjectConstPtr object) {
  if (isExprType(object->type())) {
    return object->as<Expr>()->singleTable();
  }
  return nullptr;
}

PlanObjectConstPtr Expr::singleTable() const {
  if (type() == PlanType::kColumn) {
    return as<Column>()->relation();
  }
  PlanObjectConstPtr table = nullptr;
  bool multiple = false;
  columns_.forEach([&](PlanObjectConstPtr object) {
    VELOX_CHECK_EQ(object->type(), PlanType::kColumn);
    if (!table) {
      table = object->template as<Column>()->relation();
    } else if (table != object->as<Column>()->relation()) {
      multiple = true;
    }
  });
  return multiple ? nullptr : table;
}

PlanObjectSet Expr::allTables() const {
  PlanObjectSet set;
  columns_.forEach([&](PlanObjectConstPtr object) {
    set.add(object->as<Column>()->relation());
  });
  return set;
}

PlanObjectSet allTables(PtrSpan<Expr> exprs) {
  PlanObjectSet all;
  for (auto expr : exprs) {
    auto set = expr->allTables();
    all.unionSet(set);
  }
  return all;
}

Column::Column(Name name, PlanObjectPtr relation, const Value& value)
    : Expr(PlanType::kColumn, value), name_(name), relation_(relation) {
  columns_.add(this);
  if (relation_ && relation_->type() == PlanType::kTable) {
    schemaColumn_ = relation->as<BaseTable>()->schemaTable->findColumn(name_);
    VELOX_CHECK(schemaColumn_);
  }
}

void DerivedTable::addJoinEquality(
    ExprPtr left,
    ExprPtr right,
    const ExprVector& filter,
    bool leftOptional,
    bool rightOptional,
    bool rightExists,
    bool rightNotExists) {
  auto leftTable = singleTable(left);
  auto rightTable = singleTable(right);
  for (auto& join : joins) {
    if (join->leftTable() == leftTable && join->rightTable() == rightTable) {
      join->addEquality(left, right);
      return;
    } else if (
        join->rightTable() == leftTable && join->leftTable() == rightTable) {
      join->addEquality(right, left);
      return;
    }
  }
  Declare(
      JoinEdge,
      join,
      leftTable,
      rightTable,
      filter,
      leftOptional,
      rightOptional,
      rightExists,
      rightNotExists);
  join->addEquality(left, right);
  joins.push_back(join);
}

using EdgeSet = std::unordered_set<std::pair<int32_t, int32_t>>;

bool hasEdge(const EdgeSet& edges, int32_t id1, int32_t id2) {
  if (id1 == id2) {
    return true;
  }
  auto it = edges.find(
      id1 > id2 ? std::pair<int32_t, int32_t>(id2, id1)
                : std::pair<int32_t, int32_t>(id1, id2));
  return it != edges.end();
}

void addEdge(EdgeSet& edges, int32_t id1, int32_t id2) {
  if (id1 > id2) {
    edges.insert(std::pair<int32_t, int32_t>(id2, id1));
  } else {
    edges.insert(std::pair<int32_t, int32_t>(id1, id2));
  }
}

void fillJoins(
    PlanObjectConstPtr column,
    const Equivalence& equivalence,
    EdgeSet& edges,
    DerivedTablePtr dt) {
  for (auto& other : equivalence.columns) {
    if (!hasEdge(edges, column->id(), other->id())) {
      addEdge(edges, column->id(), other->id());
      dt->addJoinEquality(
          column->as<Column>(),
          other->as<Column>(),
          {},
          false,
          false,
          false,
          false);
    }
  }
}

void DerivedTable::addImpliedJoins() {
  EdgeSet edges;
  for (auto& join : joins) {
    if (join->isInner()) {
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        if (join->leftKeys()[i]->type() == PlanType::kColumn &&
            join->rightKeys()[i]->type() == PlanType::kColumn) {
          addEdge(edges, join->leftKeys()[i]->id(), join->rightKeys()[i]->id());
        }
      }
    }
  }
  // The loop appends to 'joins', so loop over a copy.
  JoinEdgeVector joinsCopy = joins;
  for (auto& join : joinsCopy) {
    if (join->isInner()) {
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        if (join->leftKeys()[i]->type() == PlanType::kColumn &&
            join->rightKeys()[i]->type() == PlanType::kColumn) {
          auto leftEq = join->leftKeys()[i]->as<Column>()->equivalence();
          auto rightEq = join->rightKeys()[i]->as<Column>()->equivalence();
          if (rightEq && leftEq) {
            for (auto& left : leftEq->columns) {
              fillJoins(left, *rightEq, edges, this);
            }
          } else if (leftEq) {
            fillJoins(join->rightKeys()[i], *leftEq, edges, this);
          } else if (rightEq) {
            fillJoins(join->leftKeys()[i], *rightEq, edges, this);
          }
        }
      }
    }
  }
}

void DerivedTable::setStartTables() {
  startTables = tableSet;
  for (auto join : joins) {
    if (join->isNonCommutative()) {
      startTables.erase(join->rightTable());
    }
  }
}

void DerivedTable::linkTablesToJoins() {
  setStartTables();

  // All tables directly mentioned by a join link to the join. A non-inner
  // that depends on multiple left tables has no leftTable but is still linked
  // from all the tables it depends on.
  for (auto join : joins) {
    PlanObjectSet tables;
    for (auto key : join->leftKeys()) {
      tables.unionSet(key->allTables());
    }
    for (auto key : join->rightKeys()) {
      tables.unionSet(key->allTables());
    }
    if (!join->filter().empty()) {
      for (auto& conjunct : join->filter()) {
        tables.unionSet(conjunct->allTables());
      }
    }
    tables.forEachMutable([&](PlanObjectPtr table) {
      if (table->type() == PlanType::kTable) {
        table->as<BaseTable>()->addJoinedBy(join);
      } else {
        VELOX_CHECK_EQ(table->type(), PlanType::kDerivedTable);
        table->as<DerivedTable>()->addJoinedBy(join);
      }
    });
  }
}

// Returns a right exists (semijoin) with 'table' on the left and one of
// 'tables'// on the right.
JoinEdgePtr makeExists(PlanObjectConstPtr table, PlanObjectSet tables) {
  for (auto join : joinedBy(table)) {
    if (join->leftTable() == table) {
      Declare(
          JoinEdge,
          exists,
          table,
          join->rightTable(),
          {},
          false,
          false,
          true,
          false);
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        exists->addEquality(join->leftKeys()[i], join->rightKeys()[i]);
      }
      return exists;
    }
    if (join->rightTable() == table) {
      Declare(
          JoinEdge,
          exists,
          table,
          join->leftTable(),
          {},
          false,
          false,
          true,
          false);
      for (auto i = 0; i < join->leftKeys().size(); ++i) {
        exists->addEquality(join->rightKeys()[i], join->leftKeys()[i]);
      }
      return exists;
    }
  }
  VELOX_UNREACHABLE("No join to make an exists build side restriction");
}

void DerivedTable::import(
    const DerivedTable& super,
    PlanObjectConstPtr firstTable,
    const PlanObjectSet& _tables,
    const std::vector<PlanObjectSet>& existences,
    float existsFanout) {
  tableSet = _tables;
  _tables.forEach([&](auto table) { tables.push_back(table); });
  for (auto join : super.joins) {
    if (_tables.contains(join->rightTable()) && join->leftTable() &&
        _tables.contains(join->leftTable())) {
      joins.push_back(join);
    }
  }
  for (auto& exists : existences) {
    // We filter the derived table by importing reducing semijoins.
    // These are based on joins on the outer query but become
    // existences so as not to change cardinality. The reducing join
    // is against one or more tables. If more than one table, the join
    // of these tables goes into its own derived table which is joined
    // with exists to the main table(s) in the 'this'.
    importedExistences.unionSet(exists);
    std::vector<PlanObjectConstPtr, QGAllocator<PlanObjectConstPtr>>
        existsTables;
    exists.forEach([&](auto object) { existsTables.push_back(object); });
    auto existsJoin = makeExists(firstTable, exists);
    if (existsTables.size() > 1) {
      // There is a join on the right of exists. Needs its own dt.
      Declare(DerivedTable, existsDt);
      PlanObjectSet existsTableSet;
      existsTableSet.unionObjects(existsTables);
      existsDt->import(super, firstTable, existsTableSet, {});
      for (auto& k : existsJoin->rightKeys()) {
        // TODO make a column alias for the expr. this would not work if the
        // join term was not a column.
        existsDt->columns.push_back(dynamic_cast<ColumnPtr>(k));
        existsDt->exprs.push_back(k);
      }
      Declare(
          JoinEdge,
          joinWithDt,
          firstTable,
          existsDt,
          {},
          false,
          false,
          true,
          false);
      joinWithDt->setFanouts(existsFanout, 1);
      for (auto i = 0; i < existsJoin->leftKeys().size(); ++i) {
        joinWithDt->addEquality(
            existsJoin->leftKeys()[i], existsDt->columns[i]);
      }
      joins.push_back(joinWithDt);
      tables.push_back(existsDt);
      tableSet.add(existsDt);
    } else {
      joins.push_back(existsJoin);
      tables.push_back(existsTables[0]);
      tableSet.add(existsTables[0]);
    }
  }
  if (firstTable->type() == PlanType::kDerivedTable) {
    importJoinsIntoFirstDt(firstTable->as<DerivedTable>());
  } else {
    fullyImported = _tables;
  }
  linkTablesToJoins();
}

// Returns a copy of 'expr,, replacing instances of columns in 'outer' with the
// corresponding expression from 'inner'
ExprPtr
importExpr(ExprPtr expr, const ColumnVector& outer, const ExprVector& inner) {
  if (!expr) {
    return nullptr;
  }
  switch (expr->type()) {
    case PlanType::kColumn:
      for (auto i = 0; i < inner.size(); ++i) {
        if (outer[i] == expr) {
          return inner[i];
        }
      }
      return expr;
    case PlanType::kLiteral:
      return expr;
    case PlanType::kCall:
    case PlanType::kAggregate: {
      auto children = expr->children();
      std::vector<ExprPtr> newChildren(children.size());
      FunctionSet functions;
      bool anyChange = false;
      for (auto i = 0; i < children.size(); ++i) {
        newChildren[i] = importExpr(children[i]->as<Expr>(), outer, inner);
        anyChange |= newChildren[i] != children[i];
        if (newChildren[i]->isFunction()) {
          functions = functions | newChildren[i]->as<Call>()->functions();
        }
      }
      ExprPtr newCondition = nullptr;
      if (expr->type() == PlanType::kAggregate) {
        newCondition =
            importExpr(expr->as<Aggregate>()->condition(), outer, inner);
        anyChange |= newCondition != expr->as<Aggregate>()->condition();
        if (newCondition->isFunction()) {
          functions = functions | newCondition->as<Call>()->functions();
        }
      }
      if (!anyChange) {
        return expr;
      }
      ExprVector childVector;
      childVector.insert(
          childVector.begin(), newChildren.begin(), newChildren.end());
      if (expr->type() == PlanType::kCall) {
        auto call = expr->as<Call>();
        Declare(
            Call,
            copy,
            call->name(),
            call->value(),
            std::move(childVector),
            functions);
        return copy;
      } else if (expr->type() == PlanType::kAggregate) {
        auto aggregate = expr->as<Aggregate>();
        Declare(
            Aggregate,
            copy,
            aggregate->name(),
            aggregate->value(),
            std::move(childVector),
            functions,
            aggregate->isDistinct(),
            newCondition,
            aggregate->isAccumulator());
        return copy;
      }
    }
    FOLLY_FALLTHROUGH; default:
      VELOX_UNREACHABLE();
  }
}

PlanObjectConstPtr otherSide(JoinEdgePtr join, PlanObjectConstPtr side) {
  if (side == join->leftTable()) {
    return join->rightTable();
  } else if (join->rightTable() == side) {
    return join->leftTable();
  }
  return nullptr;
}

bool isProjected(PlanObjectConstPtr table, PlanObjectSet columns) {
  bool projected = false;
  columns.forEach([&](PlanObjectConstPtr column) {
    projected |= column->as<Column>()->relation() == table;
  });
  return projected;
}

// True if 'join'  has max 1 match for a row of 'side'.
bool isUnique(JoinEdgePtr join, PlanObjectConstPtr side) {
  return join->sideOf(side, true).isUnique;
}

PlanObjectConstPtr nextJoin(
    PlanObjectConstPtr start,
    const JoinEdgeVector& joins,
    PlanObjectSet columns,
    PlanObjectSet visited,
    bool& isContained) {
  for (auto& join : joins) {
    auto other = otherSide(join, start);
    if (!other) {
      continue;
    }
    if (visited.contains(other)) {
      continue;
    }
    if (!isUnique(join, other) || isProjected(other, columns)) {
      isContained = false;
    }
    return other;
  }
  return nullptr;
}

void joinChain(
    PlanObjectConstPtr start,
    const JoinEdgeVector& joins,
    PlanObjectSet columns,
    PlanObjectSet visited,
    bool& isContained,
    std::vector<PlanObjectConstPtr>& path) {
  auto next = nextJoin(start, joins, columns, visited, isContained);
  if (!next) {
    return;
  }
  visited.add(next);
  path.push_back(next);
  joinChain(next, joins, columns, visited, isContained, path);
}
JoinEdgePtr importedJoin(
    JoinEdgePtr join,
    PlanObjectConstPtr other,
    ExprPtr innerKey,
    bool isContained) {
  auto left = singleTable(innerKey);
  VELOX_CHECK(left);
  auto otherKey = join->sideOf(other).keys[0];
  Declare(
      JoinEdge, newJoin, left, other, {}, false, false, !isContained, false);
  newJoin->addEquality(innerKey, otherKey);
  return newJoin;
}

template <typename V, typename E>
void eraseFirst(V& set, E element) {
  set.erase(std::find(set.begin(), set.end(), element));
}

void DerivedTable::importJoinsIntoFirstDt(const DerivedTable* firstDt) {
  if (tables.size() == 1 && tables[0]->type() == PlanType::kDerivedTable) {
    flattenDt(tables[0]->as<DerivedTable>());
    return;
  }
  if (firstDt->limit != -1 || firstDt->orderBy) {
    return;
  }
  auto& outer = firstDt->columns;
  auto& inner = firstDt->exprs;
  PlanObjectSet projected;
  for (auto& expr : exprs) {
    projected.unionColumns(exprs);
  }

  Declare(DerivedTable, newFirst, *firstDt->as<DerivedTable>());
  newFirst->cname = firstDt->as<DerivedTable>()->cname;
  for (auto& join : joins) {
    auto other = otherSide(join, firstDt);
    if (!other) {
      continue;
    }
    auto side = join->sideOf(firstDt);
    if (side.keys.size() > 1 || !join->filter().empty()) {
      continue;
    }
    auto innerKey = importExpr(side.keys[0], outer, inner);
    if (innerKey->containsFunction(FunctionSet::kAggregate)) {
      // If the join key is an aggregate, the join can't be moved below the agg.
      continue;
    }

    PlanObjectSet visited;
    visited.add(firstDt);
    visited.add(other);
    std::vector<PlanObjectConstPtr> path;
    bool isContained = true;
    joinChain(other, joins, projected, visited, isContained, path);
    if (path.empty()) {
      newFirst->tables.push_back(other);
      newFirst->tableSet.add(other);
      newFirst->joins.push_back(
          importedJoin(join, other, innerKey, isContained));
    } else {
      Declare(DerivedTable, chainDt);
      PlanObjectSet chainSet;
      for (auto& object : path) {
        chainSet.add(object);
      }
      chainDt->import(*this, other, chainSet, {}, 1);
      newFirst->tables.push_back(chainDt);
      newFirst->tableSet.add(chainDt);
      newFirst->joins.push_back(
          importedJoin(join, other, innerKey, isContained));
    }
    if (isContained) {
      newFirst->fullyImported.add(other);
      newFirst->fullyImported.unionObjects(path);
      eraseFirst(tables, other);
      tableSet.erase(other);
      for (auto& table : path) {
        eraseFirst(tables, table);
        tableSet.erase(table);
      }
    }
  }
  tables[0] = newFirst;
  if (tables.size() == 1) {
    // All the joins and existences were added into the first derived table.
    // Flatten the first dt into 'this'.
    flattenDt(newFirst);
  } else {
    newFirst->linkTablesToJoins();
  }
}

void DerivedTable::flattenDt(const DerivedTable* dt) {
  tables = dt->tables;
  cname = dt->cname;
  tableSet = dt->tableSet;
  joins = dt->joins;
  columns = dt->columns;
  exprs = dt->exprs;
  fullyImported = dt->fullyImported;
  aggregation = dt->aggregation;
  having = dt->having;
}

std::string DerivedTable::toString() const {
  std::stringstream out;
  out << "{dt " << cname;
  for (auto& join : joins) {
    out << join->toString();
  }
  if (!conjuncts.empty()) {
    out << " where " << conjunctsToString(conjuncts);
  }
  out << "}";
  return out.str();
}

std::vector<ColumnPtr> SchemaTable::toColumns(
    const std::vector<std::string>& names) {
  std::vector<ColumnPtr> columns(names.size());
  for (auto i = 0; i < names.size(); ++i) {
    columns[i] = findColumn(name);
  }

  return columns;
}

void SchemaTable::addIndex(
    const char* name,
    float cardinality,
    int32_t numKeysUnique,
    int32_t numOrdering,
    const ColumnVector& keys,
    DistributionType distType,
    const ColumnVector& partition,
    const ColumnVector& columns) {
  Distribution distribution;
  distribution.cardinality = cardinality;
  for (auto i = 0; i < numOrdering; ++i) {
    distribution.orderType.push_back(OrderType::kAscNullsFirst);
  }
  distribution.numKeysUnique = numKeysUnique;
  appendToVector(distribution.order, keys);
  distribution.distributionType = distType;
  appendToVector(distribution.partition, partition);
  Declare(Index, index, name, this, distribution, columns);
  indices.push_back(index);
}

ColumnPtr SchemaTable::column(const std::string& name, const Value& value) {
  auto it = columns.find(toName(name));
  if (it != columns.end()) {
    return it->second;
  }
  Declare(Column, column, toName(name), nullptr, value);
  columns[toName(name)] = column;
  return column;
}

ColumnPtr SchemaTable::findColumn(const std::string& name) const {
  auto it = columns.find(toName(name));
  VELOX_CHECK(it != columns.end());
  return it->second;
}
 
Schema::Schema(const char* _name, std::vector<SchemaTablePtr> tables)
    : name(_name) {
  for (auto& table : tables) {
    tables_[table->name] = table;
  }
}

SchemaTablePtr Schema::findTable(const std::string& name) const {
  auto it = tables_.find(toName(name));
  if (it == tables_.end()) {
    VELOX_FAIL("No table {}", name);
  }
  return it->second;
}

template <typename T>
ColumnPtr findColumnByName(const T& columns, Name name) {
  for (auto column : columns) {
    if (column->type() == PlanType::kColumn &&
        column->template as<Column>()->name() == name) {
      return column->template as<Column>();
    }
  }
  return nullptr;
}

bool SchemaTable::isUnique(PtrSpan<Column> columns) const {
  for (auto index : indices) {
    auto nUnique = index->distribution().numKeysUnique;
    if (!nUnique) {
      continue;
    }
    bool unique = true;
    for (auto i = 0; i < nUnique; ++i) {
      auto part = findColumnByName(columns, index->columns()[i]->name());
      if (!part) {
        unique = false;
        break;
      }
    }
    if (unique) {
      return true;
    }
  }
  return false;
}

float combine(float card, int32_t ith, float otherCard) {
  if (ith == 0) {
    return card / otherCard;
  }
  if (otherCard > card) {
    return 1;
  }
  return card / otherCard;
}

IndexInfo SchemaTable::indexInfo(IndexPtr index, PtrSpan<Column> columns)
    const {
  IndexInfo info;
  info.index = index;
  info.scanCardinality = index->distribution().cardinality;
  info.joinCardinality = index->distribution().cardinality;
  PlanObjectSet covered;
  int32_t numCovered = 0;
  int32_t numSorting = index->distribution().orderType.size();
  int32_t numUnique = index->distribution().numKeysUnique;
  for (auto i = 0; i < numSorting || i < numUnique; ++i) {
    auto part = findColumnByName(
        columns, index->distribution().order[i]->as<Column>()->name());
    if (!part) {
      break;
    }
    ++numCovered;
    covered.add(part);
    if (i < numSorting) {
      info.scanCardinality = combine(
          info.scanCardinality,
          i,
          index->distribution().order[i]->value().cardinality);
      info.lookupKeys.push_back(part);
      info.joinCardinality = info.scanCardinality;
    } else {
      info.joinCardinality = combine(
          info.joinCardinality,
          i,
          index->distribution().order[i]->value().cardinality);
    }
    if (i == numUnique - 1) {
      info.unique = true;
    }
  }

  for (auto i = 0; i < columns.size(); ++i) {
    auto column = columns[i];
    if (covered.contains(column)) {
      continue;
    }
    auto part = findColumnByName(index->columns(), column->name());
    if (!part) {
      continue;
    }
    covered.add(column);
    ++numCovered;
    info.joinCardinality =
        combine(info.joinCardinality, numCovered, column->value().cardinality);
  }
  info.coveredColumns = std::move(covered);
  return info;
}

IndexInfo SchemaTable::indexByColumns(PtrSpan<Column> columns) const {
  // Match 'columns' against all indices. Pick the one that has the
  // longest prefix intersection with 'columns'. If 'columns' are a
  // unique combination on any index, then unique is true of the
  // result.
  IndexInfo pkInfo;
  IndexInfo best;
  bool unique = isUnique(columns);
  float bestPrediction = 0;
  for (auto iIndex = 0; iIndex < indices.size(); ++iIndex) {
    auto index = indices[iIndex];
    auto candidate = indexInfo(index, columns);
    if (iIndex == 0) {
      pkInfo = candidate;
      best = candidate;
      bestPrediction = best.joinCardinality;
      continue;
    }
    if (candidate.lookupKeys.empty()) {
      // No prefix match for secondary idex.
      continue;
    }
    // The join cardinality estimate from the longest prefix is preferred for
    // the estimate. The index with the least scan cardinality is preferred
    if (candidate.lookupKeys.size() > best.lookupKeys.size()) {
      bestPrediction = candidate.joinCardinality;
    }
    if (candidate.scanCardinality < best.scanCardinality) {
      best = candidate;
    }
  }
  best.joinCardinality = bestPrediction;
  best.unique = unique;
  return best;
}

IndexInfo joinCardinality(PlanObjectConstPtr table, PtrSpan<Column> keys) {
  if (table->type() == PlanType::kTable) {
    auto schemaTable = table->as<BaseTable>()->schemaTable;
    return schemaTable->indexByColumns(keys);
  }
  VELOX_CHECK_EQ(table->type(), PlanType::kDerivedTable);
  auto dt = table->as<DerivedTable>();
  auto distribution = dt->distribution;
  assert(distribution);
  IndexInfo result;
  result.scanCardinality = distribution->cardinality;
  const ExprVector* groupingKeys = nullptr;
  if (dt->aggregation) {
    groupingKeys = &dt->aggregation->grouping;
  }
  result.joinCardinality = result.scanCardinality;
  for (auto i = 0; i < keys.size(); ++i) {
    result.joinCardinality =
        combine(result.joinCardinality, i, keys[i]->value().cardinality);
  }
  if (groupingKeys && keys.size() >= groupingKeys->size()) {
    result.unique = true;
  }
  return result;
}

ColumnPtr IndexInfo::schemaColumn(ColumnPtr keyValue) const {
  for (auto& column : index->columns()) {
    if (column->name() == keyValue->name()) {
      return column;
    }
  }
  return nullptr;
}

// The fraction of rows of a base table selected by non-join filters. 0.2
// means 1 in 5 are selected.
float baseSelectivity(PlanObjectConstPtr object) {
  if (object->type() == PlanType::kTable) {
    return object->as<BaseTable>()->filterSelectivity;
  }
  return 1;
}

float tableCardinality(PlanObjectConstPtr table) {
  if (table->type() == PlanType::kTable) {
    return table->as<BaseTable>()
        ->schemaTable->indices[0]
        ->distribution()
        .cardinality;
  }
  VELOX_CHECK(table->type() == PlanType::kDerivedTable);
  return table->as<DerivedTable>()->distribution->cardinality;
}

void JoinEdge::guessFanout() {
  if (fanoutsFixed_) {
    return;
  }
  auto left = joinCardinality(leftTable_, toRangeCast<Column>(leftKeys_));
  auto right = joinCardinality(rightTable_, toRangeCast<Column>(rightKeys_));
  leftUnique_ = left.unique;
  rightUnique_ = right.unique;
  lrFanout_ = right.joinCardinality * baseSelectivity(rightTable_);
  rlFanout_ = left.joinCardinality * baseSelectivity(leftTable_);
  // If one side is unique, the other side is a pk to fk join, with fanout =
  // fk-table-card / pk-table-card.
  if (rightUnique_) {
    lrFanout_ = baseSelectivity(rightTable_);
    rlFanout_ = tableCardinality(leftTable_) / tableCardinality(rightTable_) *
        baseSelectivity(leftTable_);
  }
  if (leftUnique_) {
    rlFanout_ = baseSelectivity(leftTable_);
    lrFanout_ = tableCardinality(rightTable_) / tableCardinality(leftTable_) *
        baseSelectivity(rightTable_);
  }
}

bool Distribution::isSamePartition(const Distribution& other) const {
  if (!(distributionType == other.distributionType)) {
    return false;
  }
  if (isBroadcast || other.isBroadcast) {
    return true;
  }
  if (partition.size() != other.partition.size()) {
    return false;
  }
  if (partition.size() == 0) {
    // If the partitioning columns are not in the columns or if there
    // are no partitioning columns, there can be  no copartitioning.
    return false;
  }
  for (auto i = 0; i < partition.size(); ++i) {
    if (!partition[i]->sameOrEqual(*other.partition[i])) {
      return false;
    }
  }
  return true;
}

Distribution Distribution::rename(
    const ExprVector& exprs,
    const ColumnVector& names) const {
  Distribution result = *this;
  // Partitioning survives projection if all partitioning columns are projected
  // out.
  if (!replace(result.partition, exprs, names)) {
    result.partition.clear();
  }
  // Ordering survives if a prefix of the previous order continues to be
  // projected out.
  result.order.resize(prefixSize(result.order, exprs));
  replace(result.order, exprs, names);
  return result;
}

void exprsToString(const ExprVector& exprs, std::stringstream& out) {
  int32_t size = exprs.size();
  for (auto i = 0; i < size; ++i) {
    out << exprs[i]->toString() << (i < size - 1 ? ", " : "");
  }
}

std::string Distribution::toString() const {
  if (isBroadcast) {
    return "broadcast";
  }
  std::stringstream out;
  if (!partition.empty()) {
    out << "P ";
    exprsToString(partition, out);
    out << " " << distributionType.numPartitions << " ways";
  }
  if (!order.empty()) {
    out << " O ";
    exprsToString(order, out);
  }
  if (numKeysUnique && numKeysUnique >= order.size()) {
    out << " first " << numKeysUnique << " unique";
  }
  return out.str();
}

} // namespace facebook::verax
