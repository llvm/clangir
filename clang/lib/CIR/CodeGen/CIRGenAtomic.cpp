//===--- CIRGenAtomic.cpp - Emit CIR for atomic operations ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the code for emitting atomic operations.
//
//===----------------------------------------------------------------------===//

#include "Address.h"

#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "TargetInfo.h"
#include "clang/AST/ASTContext.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

namespace {
class AtomicInfo {
  CIRGenFunction &cgf;
  QualType atomicTy;
  QualType valueTy;
  uint64_t AtomicSizeInBits = 0;
  uint64_t ValueSizeInBits = 0;
  CharUnits atomicAlign;
  CharUnits valueAlign;
  TypeEvaluationKind EvaluationKind = TEK_Scalar;
  bool UseLibcall = true;
  LValue LVal;
  CIRGenBitFieldInfo bfi;
  mlir::Location loc;

public:
  AtomicInfo(CIRGenFunction &cgf, LValue &lvalue, mlir::Location l)
      : cgf(cgf), loc(l) {
    assert(!lvalue.isGlobalReg());
    ASTContext &c = cgf.getContext();
    if (lvalue.isSimple()) {
      atomicTy = lvalue.getType();
      if (auto *aTy = atomicTy->getAs<AtomicType>())
        valueTy = aTy->getValueType();
      else
        valueTy = atomicTy;
      EvaluationKind = cgf.getEvaluationKind(valueTy);

      uint64_t valueAlignInBits;
      uint64_t atomicAlignInBits;
      TypeInfo valueTi = c.getTypeInfo(valueTy);
      ValueSizeInBits = valueTi.Width;
      valueAlignInBits = valueTi.Align;

      TypeInfo atomicTi = c.getTypeInfo(atomicTy);
      AtomicSizeInBits = atomicTi.Width;
      atomicAlignInBits = atomicTi.Align;

      assert(ValueSizeInBits <= AtomicSizeInBits);
      assert(valueAlignInBits <= atomicAlignInBits);

      atomicAlign = c.toCharUnitsFromBits(atomicAlignInBits);
      valueAlign = c.toCharUnitsFromBits(valueAlignInBits);
      if (lvalue.getAlignment().isZero())
        lvalue.setAlignment(atomicAlign);

      LVal = lvalue;
    } else if (lvalue.isBitField()) {
      llvm_unreachable("NYI");
    } else if (lvalue.isVectorElt()) {
      valueTy = lvalue.getType()->castAs<VectorType>()->getElementType();
      ValueSizeInBits = c.getTypeSize(valueTy);
      atomicTy = lvalue.getType();
      AtomicSizeInBits = c.getTypeSize(atomicTy);
      atomicAlign = valueAlign = lvalue.getAlignment();
      LVal = lvalue;
    } else {
      llvm_unreachable("NYI");
    }
    UseLibcall = !c.getTargetInfo().hasBuiltinAtomic(
        AtomicSizeInBits, c.toBits(lvalue.getAlignment()));
  }

  QualType getAtomicType() const { return atomicTy; }
  QualType getValueType() const { return valueTy; }
  CharUnits getAtomicAlignment() const { return atomicAlign; }
  uint64_t getAtomicSizeInBits() const { return AtomicSizeInBits; }
  uint64_t getValueSizeInBits() const { return ValueSizeInBits; }
  TypeEvaluationKind getEvaluationKind() const { return EvaluationKind; }
  bool shouldUseLibcall() const { return UseLibcall; }
  const LValue &getAtomicLValue() const { return LVal; }
  mlir::Value getAtomicPointer() const {
    if (LVal.isSimple())
      return LVal.getPointer();
    if (LVal.isBitField())
      return LVal.getBitFieldPointer();
    else if (LVal.isVectorElt())
      return LVal.getVectorPointer();
    assert(LVal.isExtVectorElt());
    // TODO(cir): return LVal.getExtVectorPointer();
    llvm_unreachable("NYI");
  }
  Address getAtomicAddress() const {
    mlir::Type elTy;
    if (LVal.isSimple())
      elTy = LVal.getAddress().getElementType();
    else if (LVal.isBitField())
      elTy = LVal.getBitFieldAddress().getElementType();
    else if (LVal.isVectorElt())
      elTy = LVal.getVectorAddress().getElementType();
    else // TODO(cir): ElTy = LVal.getExtVectorAddress().getElementType();
      llvm_unreachable("NYI");
    return Address(getAtomicPointer(), elTy, getAtomicAlignment());
  }

  Address getAtomicAddressAsAtomicIntPointer() const {
    return castToAtomicIntPointer(getAtomicAddress());
  }

  /// Is the atomic size larger than the underlying value type?
  ///
  /// Note that the absence of padding does not mean that atomic
  /// objects are completely interchangeable with non-atomic
  /// objects: we might have promoted the alignment of a type
  /// without making it bigger.
  bool hasPadding() const { return (ValueSizeInBits != AtomicSizeInBits); }

  bool emitMemSetZeroIfNecessary() const;

  mlir::Value getAtomicSizeValue() const { llvm_unreachable("NYI"); }

  mlir::Value getScalarRValValueOrNull(RValue rVal) const;

  /// Cast the given pointer to an integer pointer suitable for atomic
  /// operations if the source.
  Address castToAtomicIntPointer(Address addr) const;

  /// If Addr is compatible with the iN that will be used for an atomic
  /// operation, bitcast it. Otherwise, create a temporary that is suitable
  /// and copy the value across.
  Address convertToAtomicIntPointer(Address addr) const;

  /// Turn an atomic-layout object into an r-value.
  RValue convertAtomicTempToRValue(Address addr, AggValueSlot resultSlot,
                                   SourceLocation loc, bool asValue) const;

  /// Converts a rvalue to integer value.
  mlir::Value convertRValueToInt(RValue rVal, bool cmpXchg = false) const;

  RValue convertIntToValueOrAtomic(mlir::Value intVal, AggValueSlot resultSlot,
                                   SourceLocation loc, bool asValue) const;

  /// Copy an atomic r-value into atomic-layout memory.
  void emitCopyIntoMemory(RValue rvalue) const;

  /// Project an l-value down to the value field.
  LValue projectValue() const {
    assert(LVal.isSimple());
    Address addr = getAtomicAddress();
    if (hasPadding())
      llvm_unreachable("NYI");

    return LValue::makeAddr(addr, getValueType(), cgf.getContext(),
                            LVal.getBaseInfo(), LVal.getTBAAInfo());
  }

  /// Emits atomic load.
  /// \returns Loaded value.
  RValue emitAtomicLoad(AggValueSlot resultSlot, SourceLocation loc,
                        bool asValue, llvm::AtomicOrdering ao, bool isVolatile);

  /// Emits atomic compare-and-exchange sequence.
  /// \param Expected Expected value.
  /// \param Desired Desired value.
  /// \param Success Atomic ordering for success operation.
  /// \param Failure Atomic ordering for failed operation.
  /// \param IsWeak true if atomic operation is weak, false otherwise.
  /// \returns Pair of values: previous value from storage (value type) and
  /// boolean flag (i1 type) with true if success and false otherwise.
  std::pair<RValue, mlir::Value>
  emitAtomicCompareExchange(RValue expected, RValue desired,
                            llvm::AtomicOrdering success =
                                llvm::AtomicOrdering::SequentiallyConsistent,
                            llvm::AtomicOrdering failure =
                                llvm::AtomicOrdering::SequentiallyConsistent,
                            bool isWeak = false);

  /// Emits atomic update.
  /// \param AO Atomic ordering.
  /// \param UpdateOp Update operation for the current lvalue.
  void emitAtomicUpdate(llvm::AtomicOrdering ao,
                        const llvm::function_ref<RValue(RValue)> &updateOp,
                        bool isVolatile);
  /// Emits atomic update.
  /// \param AO Atomic ordering.
  void emitAtomicUpdate(llvm::AtomicOrdering ao, RValue updateRVal,
                        bool isVolatile);

  /// Materialize an atomic r-value in atomic-layout memory.
  Address materializeRValue(RValue rvalue) const;

  /// Creates temp alloca for intermediate operations on atomic value.
  Address createTempAlloca() const;

private:
  bool requiresMemSetZero(mlir::Type ty) const;

  /// Emits atomic load as a libcall.
  void emitAtomicLoadLibcall(mlir::Value addForLoaded, llvm::AtomicOrdering ao,
                             bool isVolatile);
  /// Emits atomic load as LLVM instruction.
  mlir::Value emitAtomicLoadOp(llvm::AtomicOrdering ao, bool isVolatile);
  /// Emits atomic compare-and-exchange op as a libcall.
  mlir::Value emitAtomicCompareExchangeLibcall(
      mlir::Value expectedAddr, mlir::Value desiredAddr,
      llvm::AtomicOrdering success =
          llvm::AtomicOrdering::SequentiallyConsistent,
      llvm::AtomicOrdering failure =
          llvm::AtomicOrdering::SequentiallyConsistent);
  /// Emits atomic compare-and-exchange op as LLVM instruction.
  std::pair<mlir::Value, mlir::Value>
  emitAtomicCompareExchangeOp(mlir::Value expectedVal, mlir::Value desiredVal,
                              llvm::AtomicOrdering success =
                                  llvm::AtomicOrdering::SequentiallyConsistent,
                              llvm::AtomicOrdering failure =
                                  llvm::AtomicOrdering::SequentiallyConsistent,
                              bool isWeak = false);
  /// Emit atomic update as libcalls.
  void
  emitAtomicUpdateLibcall(llvm::AtomicOrdering ao,
                          const llvm::function_ref<RValue(RValue)> &updateOp,
                          bool isVolatile);
  /// Emit atomic update as LLVM instructions.
  void emitAtomicUpdateOp(llvm::AtomicOrdering ao,
                          const llvm::function_ref<RValue(RValue)> &updateOp,
                          bool isVolatile);
  /// Emit atomic update as libcalls.
  void emitAtomicUpdateLibcall(llvm::AtomicOrdering ao, RValue updateRVal,
                               bool isVolatile);
  /// Emit atomic update as LLVM instructions.
  void emitAtomicUpdateOp(llvm::AtomicOrdering ao, RValue updateRal,
                          bool isVolatile);
};
} // namespace

// This function emits any expression (scalar, complex, or aggregate)
// into a temporary alloca.
static Address buildValToTemp(CIRGenFunction &cgf, Expr *e) {
  Address declPtr = cgf.CreateMemTemp(
      e->getType(), cgf.getLoc(e->getSourceRange()), ".atomictmp");
  cgf.buildAnyExprToMem(e, declPtr, e->getType().getQualifiers(),
                        /*Init*/ true);
  return declPtr;
}

/// Does a store of the given IR type modify the full expected width?
static bool isFullSizeType(CIRGenModule &cgm, mlir::Type ty,
                           uint64_t expectedSize) {
  return (cgm.getDataLayout().getTypeStoreSize(ty) * 8 == expectedSize);
}

/// Does the atomic type require memsetting to zero before initialization?
///
/// The IR type is provided as a way of making certain queries faster.
bool AtomicInfo::requiresMemSetZero(mlir::Type ty) const {
  // If the atomic type has size padding, we definitely need a memset.
  if (hasPadding())
    return true;

  // Otherwise, do some simple heuristics to try to avoid it:
  switch (getEvaluationKind()) {
  // For scalars and complexes, check whether the store size of the
  // type uses the full size.
  case TEK_Scalar:
    return !isFullSizeType(cgf.CGM, ty, AtomicSizeInBits);
  case TEK_Complex:
    llvm_unreachable("NYI");

  // Padding in structs has an undefined bit pattern.  User beware.
  case TEK_Aggregate:
    return false;
  }
  llvm_unreachable("bad evaluation kind");
}

Address AtomicInfo::castToAtomicIntPointer(Address addr) const {
  auto intTy = mlir::dyn_cast<mlir::cir::IntType>(addr.getElementType());
  // Don't bother with int casts if the integer size is the same.
  if (intTy && intTy.getWidth() == AtomicSizeInBits)
    return addr;
  auto ty = cgf.getBuilder().getUIntNTy(AtomicSizeInBits);
  return addr.withElementType(ty);
}

Address AtomicInfo::convertToAtomicIntPointer(Address addr) const {
  auto ty = addr.getElementType();
  uint64_t sourceSizeInBits = cgf.CGM.getDataLayout().getTypeSizeInBits(ty);
  if (sourceSizeInBits != AtomicSizeInBits) {
    llvm_unreachable("NYI");
  }

  return castToAtomicIntPointer(addr);
}

Address AtomicInfo::createTempAlloca() const {
  Address tempAlloca = cgf.CreateMemTemp(
      (LVal.isBitField() && ValueSizeInBits > AtomicSizeInBits) ? valueTy
                                                                : atomicTy,
      getAtomicAlignment(), loc, "atomic-temp");
  // Cast to pointer to value type for bitfields.
  if (LVal.isBitField()) {
    llvm_unreachable("NYI");
  }
  return tempAlloca;
}

// If the value comes from a ConstOp + IntAttr, retrieve and skip a series
// of casts if necessary.
//
// FIXME(cir): figure out warning issue and move this to CIRBaseBuilder.h
static mlir::cir::IntAttr getConstOpIntAttr(mlir::Value v) {
  mlir::Operation *op = v.getDefiningOp();
  mlir::cir::IntAttr constVal;
  while (auto c = dyn_cast<mlir::cir::CastOp>(op))
    op = c.getOperand().getDefiningOp();
  if (auto c = dyn_cast<mlir::cir::ConstantOp>(op)) {
    if (mlir::isa<mlir::cir::IntType>(c.getType()))
      constVal = mlir::cast<mlir::cir::IntAttr>(c.getValue());
  }
  return constVal;
}

// Inspect a value that is the strong/weak flag for a compare-exchange.  If it
// is a constant of intergral or boolean type, set `val` to the constant's
// boolean value and return true.  Otherwise leave `val` unchanged and return
// false.
static bool isCstWeak(mlir::Value weakVal, bool &val) {
  mlir::Operation *op = weakVal.getDefiningOp();
  while (auto c = dyn_cast<mlir::cir::CastOp>(op)) {
    op = c.getOperand().getDefiningOp();
  }
  if (auto c = dyn_cast<mlir::cir::ConstantOp>(op)) {
    if (mlir::isa<mlir::cir::IntType>(c.getType())) {
      val = mlir::cast<mlir::cir::IntAttr>(c.getValue()).getUInt() != 0;
      return true;
    }
    if (mlir::isa<mlir::cir::BoolType>(c.getType())) {
      val = mlir::cast<mlir::cir::BoolAttr>(c.getValue()).getValue();
      return true;
    }
  }
  return false;
}

// Functions that help with the creation of compiler-generated switch
// statements that are used to implement non-constant memory order parameters.

// Create a new region.  Create a block within the region.  Add a "break"
// statement to the block.  Set the builder's insertion point to before the
// "break" statement.  Add the new region to the given container.
template <typename RegionsCont>
static void startRegion(mlir::OpBuilder &builder, RegionsCont &regions,
                        mlir::Location loc) {

  regions.push_back(std::make_unique<mlir::Region>());
  mlir::Region *region = regions.back().get();
  mlir::Block *block = builder.createBlock(region);
  builder.setInsertionPointToEnd(block);
  auto Break = builder.create<mlir::cir::BreakOp>(loc);
  builder.setInsertionPoint(Break);
}

// Create a "default:" label and add it to the given collection of case labels.
// Create the region that will hold the body of the "default:" block.
template <typename CaseAttrsCont, typename RegionsCont>
static void buildDefaultCase(mlir::OpBuilder &builder, CaseAttrsCont &caseAttrs,
                             RegionsCont &regions, mlir::Location loc) {

  auto *context = builder.getContext();
  auto emptyArrayAttr = builder.getArrayAttr({});
  auto defaultKind =
      mlir::cir::CaseOpKindAttr::get(context, mlir::cir::CaseOpKind::Default);
  auto defaultAttr =
      mlir::cir::CaseAttr::get(context, emptyArrayAttr, defaultKind);
  caseAttrs.push_back(defaultAttr);
  startRegion(builder, regions, loc);
}

// Create a single "case" label with the given MemOrder as its value.  Add the
// "case" label to the given collection of case labels.  Create the region that
// will hold the body of the "case" block.
template <typename CaseAttrsCont, typename RegionsCont>
static void
buildSingleMemOrderCase(mlir::OpBuilder &builder, CaseAttrsCont &caseAttrs,
                        RegionsCont &regions, mlir::Location loc,
                        mlir::Type type, mlir::cir::MemOrder order) {

  auto *context = builder.getContext();
  SmallVector<mlir::Attribute, 1> oneOrder{
      mlir::cir::IntAttr::get(type, static_cast<int>(order))};
  auto oneAttribute = builder.getArrayAttr(oneOrder);
  auto caseKind =
      mlir::cir::CaseOpKindAttr::get(context, mlir::cir::CaseOpKind::Equal);
  auto caseAttr = mlir::cir::CaseAttr::get(context, oneAttribute, caseKind);
  caseAttrs.push_back(caseAttr);
  startRegion(builder, regions, loc);
}

// Create a pair of "case" labels with the given MemOrders as their values.
// Add the combined "case" attribute to the given collection of case labels.
// Create the region that will hold the body of the "case" block.
template <typename CaseAttrsCont, typename RegionsCont>
static void buildDoubleMemOrderCase(mlir::OpBuilder &builder,
                                    CaseAttrsCont &caseAttrs,
                                    RegionsCont &regions, mlir::Location loc,
                                    mlir::Type type, mlir::cir::MemOrder order1,
                                    mlir::cir::MemOrder order2) {

  auto *context = builder.getContext();
  SmallVector<mlir::Attribute, 2> twoOrders{
      mlir::cir::IntAttr::get(type, static_cast<int>(order1)),
      mlir::cir::IntAttr::get(type, static_cast<int>(order2))};
  auto twoAttributes = builder.getArrayAttr(twoOrders);
  auto caseKind =
      mlir::cir::CaseOpKindAttr::get(context, mlir::cir::CaseOpKind::Anyof);
  auto caseAttr = mlir::cir::CaseAttr::get(context, twoAttributes, caseKind);
  caseAttrs.push_back(caseAttr);
  startRegion(builder, regions, loc);
}

static void buildAtomicCmpXchg(CIRGenFunction &cgf, AtomicExpr *e, bool isWeak,
                               Address dest, Address ptr, Address val1,
                               Address val2, uint64_t size,
                               mlir::cir::MemOrder successOrder,
                               mlir::cir::MemOrder failureOrder,
                               llvm::SyncScope::ID scope) {
  auto &builder = cgf.getBuilder();
  auto loc = cgf.getLoc(e->getSourceRange());
  auto expected = builder.createLoad(loc, val1);
  auto desired = builder.createLoad(loc, val2);
  auto boolTy = builder.getBoolTy();
  auto cmpxchg = builder.create<mlir::cir::AtomicCmpXchg>(
      loc, expected.getType(), boolTy, ptr.getPointer(), expected, desired,
      successOrder, failureOrder);
  cmpxchg.setIsVolatile(e->isVolatile());
  cmpxchg.setWeak(isWeak);

  auto cmp = builder.createNot(cmpxchg.getCmp());
  builder.create<mlir::cir::IfOp>(
      loc, cmp, false, [&](mlir::OpBuilder &, mlir::Location) {
        auto ptrTy =
            mlir::cast<mlir::cir::PointerType>(val1.getPointer().getType());
        if (val1.getElementType() != ptrTy.getPointee()) {
          val1 = val1.withPointer(builder.createPtrBitcast(
              val1.getPointer(), val1.getElementType()));
        }
        builder.createStore(loc, cmpxchg.getOld(), val1);
        builder.createYield(loc);
      });

  // Update the memory at Dest with Cmp's value.
  cgf.buildStoreOfScalar(cmpxchg.getCmp(),
                         cgf.makeAddrLValue(dest, e->getType()));
}

/// Given an ordering required on success, emit all possible cmpxchg
/// instructions to cope with the provided (but possibly only dynamically known)
/// FailureOrder.
static void buildAtomicCmpXchgFailureSet(
    CIRGenFunction &cgf, AtomicExpr *e, bool isWeak, Address dest, Address ptr,
    Address val1, Address val2, mlir::Value failureOrderVal, uint64_t size,
    mlir::cir::MemOrder successOrder, llvm::SyncScope::ID scope) {

  mlir::cir::MemOrder failureOrder;
  if (auto ordAttr = getConstOpIntAttr(failureOrderVal)) {
    // We should not ever get to a case where the ordering isn't a valid CABI
    // value, but it's hard to enforce that in general.
    auto ord = ordAttr.getUInt();
    if (!mlir::cir::isValidCIRAtomicOrderingCABI(ord)) {
      failureOrder = mlir::cir::MemOrder::Relaxed;
    } else {
      switch ((mlir::cir::MemOrder)ord) {
      case mlir::cir::MemOrder::Relaxed:
        // 31.7.2.18: "The failure argument shall not be memory_order_release
        // nor memory_order_acq_rel". Fallback to monotonic.
      case mlir::cir::MemOrder::Release:
      case mlir::cir::MemOrder::AcquireRelease:
        failureOrder = mlir::cir::MemOrder::Relaxed;
        break;
      case mlir::cir::MemOrder::Consume:
      case mlir::cir::MemOrder::Acquire:
        failureOrder = mlir::cir::MemOrder::Acquire;
        break;
      case mlir::cir::MemOrder::SequentiallyConsistent:
        failureOrder = mlir::cir::MemOrder::SequentiallyConsistent;
        break;
      }
    }
    // Prior to c++17, "the failure argument shall be no stronger than the
    // success argument". This condition has been lifted and the only
    // precondition is 31.7.2.18. Effectively treat this as a DR and skip
    // language version checks.
    buildAtomicCmpXchg(cgf, e, isWeak, dest, ptr, val1, val2, size,
                       successOrder, failureOrder, scope);
    return;
  }

  // The failure memory order is not a compile-time value. The CIR atomic ops
  // can't handle a runtime value; all memory orders must be hard coded.
  // Generate a "switch" statement that converts the runtime value into a
  // compile-time value.
  cgf.getBuilder().create<mlir::cir::SwitchOp>(
      failureOrderVal.getLoc(), failureOrderVal,
      [&](mlir::OpBuilder &builder, mlir::Location loc,
          mlir::OperationState &os) {
        SmallVector<mlir::Attribute, 3> caseAttrs;
        SmallVector<std::unique_ptr<mlir::Region>, 3> regions;

        // default:
        // Unsupported memory orders get generated as memory_order_relaxed,
        // because there is no practical way to report an error at runtime.
        buildDefaultCase(builder, caseAttrs, regions, loc);
        buildAtomicCmpXchg(cgf, e, isWeak, dest, ptr, val1, val2, size,
                           successOrder, mlir::cir::MemOrder::Relaxed, scope);

        // case consume:
        // case acquire:
        // memory_order_consume is not implemented and always falls back to
        // memory_order_acquire
        buildDoubleMemOrderCase(
            builder, caseAttrs, regions, loc, failureOrderVal.getType(),
            mlir::cir::MemOrder::Consume, mlir::cir::MemOrder::Acquire);
        buildAtomicCmpXchg(cgf, e, isWeak, dest, ptr, val1, val2, size,
                           successOrder, mlir::cir::MemOrder::Acquire, scope);

        // A failed compare-exchange is a read-only operation.  So
        // memory_order_release and memory_order_acq_rel are not supported for
        // the failure memory order.  They fall back to memory_order_relaxed.

        // case seq_cst:
        buildSingleMemOrderCase(builder, caseAttrs, regions, loc,
                                failureOrderVal.getType(),
                                mlir::cir::MemOrder::SequentiallyConsistent);
        buildAtomicCmpXchg(cgf, e, isWeak, dest, ptr, val1, val2, size,
                           successOrder,
                           mlir::cir::MemOrder::SequentiallyConsistent, scope);

        os.addRegions(regions);
        os.addAttribute("cases", builder.getArrayAttr(caseAttrs));
      });
}

static void buildAtomicOp(CIRGenFunction &cgf, AtomicExpr *e, Address dest,
                          Address ptr, Address val1, Address val2,
                          mlir::Value isWeak, mlir::Value failureOrder,
                          uint64_t size, mlir::cir::MemOrder order,
                          uint8_t scope) {
  assert(!MissingFeatures::syncScopeID());
  StringRef Op;

  auto &builder = cgf.getBuilder();
  auto loc = cgf.getLoc(e->getSourceRange());
  auto orderAttr = mlir::cir::MemOrderAttr::get(builder.getContext(), order);
  mlir::cir::AtomicFetchKindAttr fetchAttr;
  bool fetchFirst = true;

  switch (e->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
  case AtomicExpr::AO__opencl_atomic_init:
    llvm_unreachable("Already handled!");

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
    buildAtomicCmpXchgFailureSet(cgf, e, false, dest, ptr, val1, val2,
                                 failureOrder, size, order, scope);
    return;
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
  case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
    llvm_unreachable("NYI");
    return;
  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n:
  case AtomicExpr::AO__scoped_atomic_compare_exchange:
  case AtomicExpr::AO__scoped_atomic_compare_exchange_n: {
    bool weakVal;
    if (isCstWeak(isWeak, weakVal)) {
      buildAtomicCmpXchgFailureSet(cgf, e, weakVal, dest, ptr, val1, val2,
                                   failureOrder, size, order, scope);
    } else {
      llvm_unreachable("NYI");
    }
    return;
  }
  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__opencl_atomic_load:
  case AtomicExpr::AO__hip_atomic_load:
  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__atomic_load:
  case AtomicExpr::AO__scoped_atomic_load_n:
  case AtomicExpr::AO__scoped_atomic_load: {
    auto *load = builder.createLoad(loc, ptr).getDefiningOp();
    // FIXME(cir): add scope information.
    assert(!MissingFeatures::syncScopeID());
    load->setAttr("mem_order", orderAttr);
    if (e->isVolatile())
      load->setAttr("is_volatile", mlir::UnitAttr::get(builder.getContext()));

    // TODO(cir): this logic should be part of createStore, but doing so
    // currently breaks CodeGen/union.cpp and CodeGen/union.cpp.
    auto ptrTy =
        mlir::cast<mlir::cir::PointerType>(dest.getPointer().getType());
    if (dest.getElementType() != ptrTy.getPointee()) {
      dest = dest.withPointer(
          builder.createPtrBitcast(dest.getPointer(), dest.getElementType()));
    }
    builder.createStore(loc, load->getResult(0), dest);
    return;
  }

  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__opencl_atomic_store:
  case AtomicExpr::AO__hip_atomic_store:
  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__scoped_atomic_store:
  case AtomicExpr::AO__scoped_atomic_store_n: {
    auto loadVal1 = builder.createLoad(loc, val1);
    // FIXME(cir): add scope information.
    assert(!MissingFeatures::syncScopeID());
    builder.createStore(loc, loadVal1, ptr, e->isVolatile(),
                        /*alignment=*/mlir::IntegerAttr{}, orderAttr);
    return;
  }

  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__hip_atomic_exchange:
  case AtomicExpr::AO__opencl_atomic_exchange:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange_n:
  case AtomicExpr::AO__scoped_atomic_exchange:
    Op = mlir::cir::AtomicXchg::getOperationName();
    break;

  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__scoped_atomic_add_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__scoped_atomic_fetch_add:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::Add);
    break;

  case AtomicExpr::AO__atomic_sub_fetch:
  case AtomicExpr::AO__scoped_atomic_sub_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__scoped_atomic_fetch_sub:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::Sub);
    break;

  case AtomicExpr::AO__atomic_min_fetch:
  case AtomicExpr::AO__scoped_atomic_min_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_min:
  case AtomicExpr::AO__hip_atomic_fetch_min:
  case AtomicExpr::AO__opencl_atomic_fetch_min:
  case AtomicExpr::AO__atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_min:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::Min);
    break;

  case AtomicExpr::AO__atomic_max_fetch:
  case AtomicExpr::AO__scoped_atomic_max_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_max:
  case AtomicExpr::AO__hip_atomic_fetch_max:
  case AtomicExpr::AO__opencl_atomic_fetch_max:
  case AtomicExpr::AO__atomic_fetch_max:
  case AtomicExpr::AO__scoped_atomic_fetch_max:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::Max);
    break;

  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__scoped_atomic_and_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__hip_atomic_fetch_and:
  case AtomicExpr::AO__opencl_atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__scoped_atomic_fetch_and:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::And);
    break;

  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__scoped_atomic_or_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__hip_atomic_fetch_or:
  case AtomicExpr::AO__opencl_atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__scoped_atomic_fetch_or:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::Or);
    break;

  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__scoped_atomic_xor_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__hip_atomic_fetch_xor:
  case AtomicExpr::AO__opencl_atomic_fetch_xor:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__scoped_atomic_fetch_xor:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::Xor);
    break;

  case AtomicExpr::AO__atomic_nand_fetch:
  case AtomicExpr::AO__scoped_atomic_nand_fetch:
    fetchFirst = false;
    [[fallthrough]];
  case AtomicExpr::AO__c11_atomic_fetch_nand:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__scoped_atomic_fetch_nand:
    Op = mlir::cir::AtomicFetch::getOperationName();
    fetchAttr = mlir::cir::AtomicFetchKindAttr::get(
        builder.getContext(), mlir::cir::AtomicFetchKind::Nand);
    break;
  }

  assert(!Op.empty() && "expected operation name to build");
  auto loadVal1 = builder.createLoad(loc, val1);

  SmallVector<mlir::Value> atomicOperands = {ptr.getPointer(), loadVal1};
  SmallVector<mlir::Type> atomicResTys = {loadVal1.getType()};
  auto *rmwi = builder.create(loc, builder.getStringAttr(Op), atomicOperands,
                              atomicResTys, {});

  if (fetchAttr)
    rmwi->setAttr("binop", fetchAttr);
  rmwi->setAttr("mem_order", orderAttr);
  if (e->isVolatile())
    rmwi->setAttr("is_volatile", mlir::UnitAttr::get(builder.getContext()));
  if (fetchFirst && Op == mlir::cir::AtomicFetch::getOperationName())
    rmwi->setAttr("fetch_first", mlir::UnitAttr::get(builder.getContext()));

  auto result = rmwi->getResult(0);

  // TODO(cir): this logic should be part of createStore, but doing so currently
  // breaks CodeGen/union.cpp and CodeGen/union.cpp.
  auto ptrTy = mlir::cast<mlir::cir::PointerType>(dest.getPointer().getType());
  if (dest.getElementType() != ptrTy.getPointee()) {
    dest = dest.withPointer(
        builder.createPtrBitcast(dest.getPointer(), dest.getElementType()));
  }
  builder.createStore(loc, result, dest);
}

static RValue buildAtomicLibcall(CIRGenFunction &cgf, StringRef fnName,
                                 QualType resultType, CallArgList &args) {
  [[maybe_unused]] const CIRGenFunctionInfo &fnInfo =
      cgf.CGM.getTypes().arrangeBuiltinFunctionCall(resultType, args);
  [[maybe_unused]] auto fnTy = cgf.CGM.getTypes().GetFunctionType(fnInfo);
  llvm_unreachable("NYI");
}

static void buildAtomicOp(CIRGenFunction &cgf, AtomicExpr *expr, Address dest,
                          Address ptr, Address val1, Address val2,
                          mlir::Value isWeak, mlir::Value failureOrder,
                          uint64_t size, mlir::cir::MemOrder order,
                          mlir::Value scope) {
  auto scopeModel = expr->getScopeModel();

  // LLVM atomic instructions always have synch scope. If clang atomic
  // expression has no scope operand, use default LLVM synch scope.
  if (!scopeModel) {
    assert(!MissingFeatures::syncScopeID());
    buildAtomicOp(cgf, expr, dest, ptr, val1, val2, isWeak, failureOrder, size,
                  order, /*FIXME(cir): LLVM default scope*/ 1);
    return;
  }

  // Handle constant scope.
  if (getConstOpIntAttr(scope)) {
    assert(!MissingFeatures::syncScopeID());
    llvm_unreachable("NYI");
    return;
  }

  // Handle non-constant scope.
  llvm_unreachable("NYI");
}

RValue CIRGenFunction::buildAtomicExpr(AtomicExpr *e) {
  QualType atomicTy = e->getPtr()->getType()->getPointeeType();
  QualType memTy = atomicTy;
  if (const AtomicType *at = atomicTy->getAs<AtomicType>())
    memTy = at->getValueType();
  mlir::Value isWeak = nullptr, orderFail = nullptr;

  Address val1 = Address::invalid();
  Address val2 = Address::invalid();
  Address dest = Address::invalid();
  Address ptr = buildPointerWithAlignment(e->getPtr());

  if (e->getOp() == AtomicExpr::AO__c11_atomic_init ||
      e->getOp() == AtomicExpr::AO__opencl_atomic_init) {
    LValue lvalue = makeAddrLValue(ptr, atomicTy);
    buildAtomicInit(e->getVal1(), lvalue);
    return RValue::get(nullptr);
  }

  auto tInfo = getContext().getTypeInfoInChars(atomicTy);
  uint64_t size = tInfo.Width.getQuantity();
  unsigned maxInlineWidthInBits = getTarget().getMaxAtomicInlineWidth();

  CharUnits maxInlineWidth =
      getContext().toCharUnitsFromBits(maxInlineWidthInBits);
  DiagnosticsEngine &diags = CGM.getDiags();
  bool misaligned = (ptr.getAlignment() % tInfo.Width) != 0;
  bool oversized = getContext().toBits(tInfo.Width) > maxInlineWidthInBits;
  if (misaligned) {
    diags.Report(e->getBeginLoc(), diag::warn_atomic_op_misaligned)
        << (int)tInfo.Width.getQuantity()
        << (int)ptr.getAlignment().getQuantity();
  }
  if (oversized) {
    diags.Report(e->getBeginLoc(), diag::warn_atomic_op_oversized)
        << (int)tInfo.Width.getQuantity() << (int)maxInlineWidth.getQuantity();
  }

  auto order = buildScalarExpr(e->getOrder());
  auto scope = e->getScopeModel() ? buildScalarExpr(e->getScope()) : nullptr;
  bool shouldCastToIntPtrTy = true;

  switch (e->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
  case AtomicExpr::AO__opencl_atomic_init:
    llvm_unreachable("Already handled above with EmitAtomicInit!");

  case AtomicExpr::AO__atomic_load_n:
  case AtomicExpr::AO__scoped_atomic_load_n:
  case AtomicExpr::AO__c11_atomic_load:
  case AtomicExpr::AO__opencl_atomic_load:
  case AtomicExpr::AO__hip_atomic_load:
    break;

  case AtomicExpr::AO__atomic_load:
  case AtomicExpr::AO__scoped_atomic_load:
    dest = buildPointerWithAlignment(e->getVal1());
    break;

  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__scoped_atomic_store:
    val1 = buildPointerWithAlignment(e->getVal1());
    break;

  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange:
    val1 = buildPointerWithAlignment(e->getVal1());
    dest = buildPointerWithAlignment(e->getVal2());
    break;

  case AtomicExpr::AO__atomic_compare_exchange:
  case AtomicExpr::AO__atomic_compare_exchange_n:
  case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
  case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
  case AtomicExpr::AO__scoped_atomic_compare_exchange:
  case AtomicExpr::AO__scoped_atomic_compare_exchange_n:
    val1 = buildPointerWithAlignment(e->getVal1());
    if (e->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
        e->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange)
      val2 = buildPointerWithAlignment(e->getVal2());
    else
      val2 = buildValToTemp(*this, e->getVal2());
    orderFail = buildScalarExpr(e->getOrderFail());
    if (e->getOp() == AtomicExpr::AO__atomic_compare_exchange_n ||
        e->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
        e->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange_n ||
        e->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange) {
      isWeak = buildScalarExpr(e->getWeak());
    }
    break;

  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
    if (memTy->isPointerType()) {
      llvm_unreachable("NYI");
    }
    [[fallthrough]];
  case AtomicExpr::AO__atomic_fetch_add:
  case AtomicExpr::AO__atomic_fetch_max:
  case AtomicExpr::AO__atomic_fetch_min:
  case AtomicExpr::AO__atomic_fetch_sub:
  case AtomicExpr::AO__atomic_add_fetch:
  case AtomicExpr::AO__atomic_max_fetch:
  case AtomicExpr::AO__atomic_min_fetch:
  case AtomicExpr::AO__atomic_sub_fetch:
  case AtomicExpr::AO__c11_atomic_fetch_max:
  case AtomicExpr::AO__c11_atomic_fetch_min:
  case AtomicExpr::AO__opencl_atomic_fetch_max:
  case AtomicExpr::AO__opencl_atomic_fetch_min:
  case AtomicExpr::AO__hip_atomic_fetch_max:
  case AtomicExpr::AO__hip_atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_add:
  case AtomicExpr::AO__scoped_atomic_fetch_max:
  case AtomicExpr::AO__scoped_atomic_fetch_min:
  case AtomicExpr::AO__scoped_atomic_fetch_sub:
  case AtomicExpr::AO__scoped_atomic_add_fetch:
  case AtomicExpr::AO__scoped_atomic_max_fetch:
  case AtomicExpr::AO__scoped_atomic_min_fetch:
  case AtomicExpr::AO__scoped_atomic_sub_fetch:
    shouldCastToIntPtrTy = !memTy->isFloatingType();
    [[fallthrough]];

  case AtomicExpr::AO__atomic_fetch_and:
  case AtomicExpr::AO__atomic_fetch_nand:
  case AtomicExpr::AO__atomic_fetch_or:
  case AtomicExpr::AO__atomic_fetch_xor:
  case AtomicExpr::AO__atomic_and_fetch:
  case AtomicExpr::AO__atomic_nand_fetch:
  case AtomicExpr::AO__atomic_or_fetch:
  case AtomicExpr::AO__atomic_xor_fetch:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__atomic_exchange_n:
  case AtomicExpr::AO__c11_atomic_fetch_and:
  case AtomicExpr::AO__c11_atomic_fetch_nand:
  case AtomicExpr::AO__c11_atomic_fetch_or:
  case AtomicExpr::AO__c11_atomic_fetch_xor:
  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__c11_atomic_exchange:
  case AtomicExpr::AO__hip_atomic_fetch_and:
  case AtomicExpr::AO__hip_atomic_fetch_or:
  case AtomicExpr::AO__hip_atomic_fetch_xor:
  case AtomicExpr::AO__hip_atomic_store:
  case AtomicExpr::AO__hip_atomic_exchange:
  case AtomicExpr::AO__opencl_atomic_fetch_and:
  case AtomicExpr::AO__opencl_atomic_fetch_or:
  case AtomicExpr::AO__opencl_atomic_fetch_xor:
  case AtomicExpr::AO__opencl_atomic_store:
  case AtomicExpr::AO__opencl_atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_fetch_and:
  case AtomicExpr::AO__scoped_atomic_fetch_nand:
  case AtomicExpr::AO__scoped_atomic_fetch_or:
  case AtomicExpr::AO__scoped_atomic_fetch_xor:
  case AtomicExpr::AO__scoped_atomic_and_fetch:
  case AtomicExpr::AO__scoped_atomic_nand_fetch:
  case AtomicExpr::AO__scoped_atomic_or_fetch:
  case AtomicExpr::AO__scoped_atomic_xor_fetch:
  case AtomicExpr::AO__scoped_atomic_store_n:
  case AtomicExpr::AO__scoped_atomic_exchange_n:
    val1 = buildValToTemp(*this, e->getVal1());
    break;
  }

  QualType rValTy = e->getType().getUnqualifiedType();

  // The inlined atomics only function on iN types, where N is a power of 2. We
  // need to make sure (via temporaries if necessary) that all incoming values
  // are compatible.
  LValue atomicVal = makeAddrLValue(ptr, atomicTy);
  AtomicInfo atomics(*this, atomicVal, getLoc(e->getSourceRange()));

  if (shouldCastToIntPtrTy) {
    ptr = atomics.castToAtomicIntPointer(ptr);
    if (val1.isValid())
      val1 = atomics.convertToAtomicIntPointer(val1);
    if (val2.isValid())
      val2 = atomics.convertToAtomicIntPointer(val2);
  }
  if (dest.isValid()) {
    if (shouldCastToIntPtrTy)
      dest = atomics.castToAtomicIntPointer(dest);
  } else if (e->isCmpXChg())
    dest = CreateMemTemp(rValTy, getLoc(e->getSourceRange()), "cmpxchg.bool");
  else if (!rValTy->isVoidType()) {
    dest = atomics.createTempAlloca();
    if (shouldCastToIntPtrTy)
      dest = atomics.castToAtomicIntPointer(dest);
  }

  bool powerOf2Size = (size & (size - 1)) == 0;
  bool useLibcall = !powerOf2Size || (size > 16);

  // For atomics larger than 16 bytes, emit a libcall from the frontend. This
  // avoids the overhead of dealing with excessively-large value types in IR.
  // Non-power-of-2 values also lower to libcall here, as they are not currently
  // permitted in IR instructions (although that constraint could be relaxed in
  // the future). For other cases where a libcall is required on a given
  // platform, we let the backend handle it (this includes handling for all of
  // the size-optimized libcall variants, which are only valid up to 16 bytes.)
  //
  // See: https://llvm.org/docs/Atomics.html#libcalls-atomic
  if (useLibcall) {
    CallArgList args;
    // For non-optimized library calls, the size is the first parameter.
    args.add(RValue::get(builder.getConstInt(getLoc(e->getSourceRange()),
                                             SizeTy, size)),
             getContext().getSizeType());

    // The atomic address is the second parameter.
    // The OpenCL atomic library functions only accept pointer arguments to
    // generic address space.
    auto castToGenericAddrSpace = [&](mlir::Value v, QualType pt) {
      if (!e->isOpenCL())
        return v;
      llvm_unreachable("NYI");
    };

    args.add(RValue::get(castToGenericAddrSpace(ptr.emitRawPointer(),
                                                e->getPtr()->getType())),
             getContext().VoidPtrTy);

    // The next 1-3 parameters are op-dependent.
    std::string libCallName;
    QualType retTy;
    bool haveRetTy = false;
    switch (e->getOp()) {
    case AtomicExpr::AO__c11_atomic_init:
    case AtomicExpr::AO__opencl_atomic_init:
      llvm_unreachable("Already handled!");

    // There is only one libcall for compare an exchange, because there is no
    // optimisation benefit possible from a libcall version of a weak compare
    // and exchange.
    // bool __atomic_compare_exchange(size_t size, void *mem, void *expected,
    //                                void *desired, int success, int failure)
    case AtomicExpr::AO__atomic_compare_exchange:
    case AtomicExpr::AO__atomic_compare_exchange_n:
    case AtomicExpr::AO__c11_atomic_compare_exchange_weak:
    case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
    case AtomicExpr::AO__hip_atomic_compare_exchange_weak:
    case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
    case AtomicExpr::AO__opencl_atomic_compare_exchange_weak:
    case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
    case AtomicExpr::AO__scoped_atomic_compare_exchange:
    case AtomicExpr::AO__scoped_atomic_compare_exchange_n:
      libCallName = "__atomic_compare_exchange";
      llvm_unreachable("NYI");
      break;
    // void __atomic_exchange(size_t size, void *mem, void *val, void *return,
    //                        int order)
    case AtomicExpr::AO__atomic_exchange:
    case AtomicExpr::AO__atomic_exchange_n:
    case AtomicExpr::AO__c11_atomic_exchange:
    case AtomicExpr::AO__hip_atomic_exchange:
    case AtomicExpr::AO__opencl_atomic_exchange:
    case AtomicExpr::AO__scoped_atomic_exchange:
    case AtomicExpr::AO__scoped_atomic_exchange_n:
      libCallName = "__atomic_exchange";
      llvm_unreachable("NYI");
      break;
    // void __atomic_store(size_t size, void *mem, void *val, int order)
    case AtomicExpr::AO__atomic_store:
    case AtomicExpr::AO__atomic_store_n:
    case AtomicExpr::AO__c11_atomic_store:
    case AtomicExpr::AO__hip_atomic_store:
    case AtomicExpr::AO__opencl_atomic_store:
    case AtomicExpr::AO__scoped_atomic_store:
    case AtomicExpr::AO__scoped_atomic_store_n:
      libCallName = "__atomic_store";
      llvm_unreachable("NYI");
      break;
    // void __atomic_load(size_t size, void *mem, void *return, int order)
    case AtomicExpr::AO__atomic_load:
    case AtomicExpr::AO__atomic_load_n:
    case AtomicExpr::AO__c11_atomic_load:
    case AtomicExpr::AO__hip_atomic_load:
    case AtomicExpr::AO__opencl_atomic_load:
    case AtomicExpr::AO__scoped_atomic_load:
    case AtomicExpr::AO__scoped_atomic_load_n:
      libCallName = "__atomic_load";
      break;
    case AtomicExpr::AO__atomic_add_fetch:
    case AtomicExpr::AO__scoped_atomic_add_fetch:
    case AtomicExpr::AO__atomic_fetch_add:
    case AtomicExpr::AO__c11_atomic_fetch_add:
    case AtomicExpr::AO__hip_atomic_fetch_add:
    case AtomicExpr::AO__opencl_atomic_fetch_add:
    case AtomicExpr::AO__scoped_atomic_fetch_add:
    case AtomicExpr::AO__atomic_and_fetch:
    case AtomicExpr::AO__scoped_atomic_and_fetch:
    case AtomicExpr::AO__atomic_fetch_and:
    case AtomicExpr::AO__c11_atomic_fetch_and:
    case AtomicExpr::AO__hip_atomic_fetch_and:
    case AtomicExpr::AO__opencl_atomic_fetch_and:
    case AtomicExpr::AO__scoped_atomic_fetch_and:
    case AtomicExpr::AO__atomic_or_fetch:
    case AtomicExpr::AO__scoped_atomic_or_fetch:
    case AtomicExpr::AO__atomic_fetch_or:
    case AtomicExpr::AO__c11_atomic_fetch_or:
    case AtomicExpr::AO__hip_atomic_fetch_or:
    case AtomicExpr::AO__opencl_atomic_fetch_or:
    case AtomicExpr::AO__scoped_atomic_fetch_or:
    case AtomicExpr::AO__atomic_sub_fetch:
    case AtomicExpr::AO__scoped_atomic_sub_fetch:
    case AtomicExpr::AO__atomic_fetch_sub:
    case AtomicExpr::AO__c11_atomic_fetch_sub:
    case AtomicExpr::AO__hip_atomic_fetch_sub:
    case AtomicExpr::AO__opencl_atomic_fetch_sub:
    case AtomicExpr::AO__scoped_atomic_fetch_sub:
    case AtomicExpr::AO__atomic_xor_fetch:
    case AtomicExpr::AO__scoped_atomic_xor_fetch:
    case AtomicExpr::AO__atomic_fetch_xor:
    case AtomicExpr::AO__c11_atomic_fetch_xor:
    case AtomicExpr::AO__hip_atomic_fetch_xor:
    case AtomicExpr::AO__opencl_atomic_fetch_xor:
    case AtomicExpr::AO__scoped_atomic_fetch_xor:
    case AtomicExpr::AO__atomic_nand_fetch:
    case AtomicExpr::AO__atomic_fetch_nand:
    case AtomicExpr::AO__c11_atomic_fetch_nand:
    case AtomicExpr::AO__scoped_atomic_fetch_nand:
    case AtomicExpr::AO__scoped_atomic_nand_fetch:
    case AtomicExpr::AO__atomic_min_fetch:
    case AtomicExpr::AO__atomic_fetch_min:
    case AtomicExpr::AO__c11_atomic_fetch_min:
    case AtomicExpr::AO__hip_atomic_fetch_min:
    case AtomicExpr::AO__opencl_atomic_fetch_min:
    case AtomicExpr::AO__scoped_atomic_fetch_min:
    case AtomicExpr::AO__scoped_atomic_min_fetch:
    case AtomicExpr::AO__atomic_max_fetch:
    case AtomicExpr::AO__atomic_fetch_max:
    case AtomicExpr::AO__c11_atomic_fetch_max:
    case AtomicExpr::AO__hip_atomic_fetch_max:
    case AtomicExpr::AO__opencl_atomic_fetch_max:
    case AtomicExpr::AO__scoped_atomic_fetch_max:
    case AtomicExpr::AO__scoped_atomic_max_fetch:
      llvm_unreachable("Integral atomic operations always become atomicrmw!");
    }

    if (e->isOpenCL()) {
      libCallName =
          std::string("__opencl") + StringRef(libCallName).drop_front(1).str();
    }
    // By default, assume we return a value of the atomic type.
    if (!haveRetTy) {
      llvm_unreachable("NYI");
    }
    // Order is always the last parameter.
    args.add(RValue::get(order), getContext().IntTy);
    if (e->isOpenCL()) {
      llvm_unreachable("NYI");
    }

    [[maybe_unused]] RValue res =
        buildAtomicLibcall(*this, libCallName, retTy, args);
    // The value is returned directly from the libcall.
    if (e->isCmpXChg()) {
      llvm_unreachable("NYI");
    }

    if (rValTy->isVoidType()) {
      llvm_unreachable("NYI");
    }

    llvm_unreachable("NYI");
  }

  [[maybe_unused]] bool isStore =
      e->getOp() == AtomicExpr::AO__c11_atomic_store ||
      e->getOp() == AtomicExpr::AO__opencl_atomic_store ||
      e->getOp() == AtomicExpr::AO__hip_atomic_store ||
      e->getOp() == AtomicExpr::AO__atomic_store ||
      e->getOp() == AtomicExpr::AO__atomic_store_n ||
      e->getOp() == AtomicExpr::AO__scoped_atomic_store ||
      e->getOp() == AtomicExpr::AO__scoped_atomic_store_n;
  [[maybe_unused]] bool isLoad =
      e->getOp() == AtomicExpr::AO__c11_atomic_load ||
      e->getOp() == AtomicExpr::AO__opencl_atomic_load ||
      e->getOp() == AtomicExpr::AO__hip_atomic_load ||
      e->getOp() == AtomicExpr::AO__atomic_load ||
      e->getOp() == AtomicExpr::AO__atomic_load_n ||
      e->getOp() == AtomicExpr::AO__scoped_atomic_load ||
      e->getOp() == AtomicExpr::AO__scoped_atomic_load_n;

  if (auto ordAttr = getConstOpIntAttr(order)) {
    // We should not ever get to a case where the ordering isn't a valid CABI
    // value, but it's hard to enforce that in general.
    auto ord = ordAttr.getUInt();
    if (mlir::cir::isValidCIRAtomicOrderingCABI(ord)) {
      switch ((mlir::cir::MemOrder)ord) {
      case mlir::cir::MemOrder::Relaxed:
        buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail, size,
                      mlir::cir::MemOrder::Relaxed, scope);
        break;
      case mlir::cir::MemOrder::Consume:
      case mlir::cir::MemOrder::Acquire:
        if (isStore)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail, size,
                      mlir::cir::MemOrder::Acquire, scope);
        break;
      case mlir::cir::MemOrder::Release:
        if (isLoad)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail, size,
                      mlir::cir::MemOrder::Release, scope);
        break;
      case mlir::cir::MemOrder::AcquireRelease:
        if (isLoad || isStore)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail, size,
                      mlir::cir::MemOrder::AcquireRelease, scope);
        break;
      case mlir::cir::MemOrder::SequentiallyConsistent:
        buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail, size,
                      mlir::cir::MemOrder::SequentiallyConsistent, scope);
        break;
      }
    }
    if (rValTy->isVoidType())
      return RValue::get(nullptr);

    return convertTempToRValue(dest.withElementType(convertTypeForMem(rValTy)),
                               rValTy, e->getExprLoc());
  }

  // The memory order is not known at compile-time.  The atomic operations
  // can't handle runtime memory orders; the memory order must be hard coded.
  // Generate a "switch" statement that converts a runtime value into a
  // compile-time value.
  builder.create<mlir::cir::SwitchOp>(
      order.getLoc(), order,
      [&](mlir::OpBuilder &builder, mlir::Location loc,
          mlir::OperationState &os) {
        llvm::SmallVector<mlir::Attribute, 6> caseAttrs;
        llvm::SmallVector<std::unique_ptr<mlir::Region>, 6> regions;

        // default:
        // Use memory_order_relaxed for relaxed operations and for any memory
        // order value that is not supported.  There is no good way to report
        // an unsupported memory order at runtime, hence the fallback to
        // memory_order_relaxed.
        buildDefaultCase(builder, caseAttrs, regions, loc);
        buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail, size,
                      mlir::cir::MemOrder::Relaxed, scope);

        if (!isStore) {
          // case consume:
          // case acquire:
          // memory_order_consume is not implemented; it is always treated like
          // memory_order_acquire.  These memory orders are not valid for
          // write-only operations.
          buildDoubleMemOrderCase(builder, caseAttrs, regions, loc,
                                  order.getType(), mlir::cir::MemOrder::Consume,
                                  mlir::cir::MemOrder::Acquire);
          buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail,
                        size, mlir::cir::MemOrder::Acquire, scope);
        }

        if (!isLoad) {
          // case release:
          // memory_order_release is not valid for read-only operations.
          buildSingleMemOrderCase(builder, caseAttrs, regions, loc,
                                  order.getType(),
                                  mlir::cir::MemOrder::Release);
          buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail,
                        size, mlir::cir::MemOrder::Release, scope);
        }

        if (!isLoad && !isStore) {
          // case acq_rel:
          // memory_order_acq_rel is only valid for read-write operations.
          buildSingleMemOrderCase(builder, caseAttrs, regions, loc,
                                  order.getType(),
                                  mlir::cir::MemOrder::AcquireRelease);
          buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail,
                        size, mlir::cir::MemOrder::AcquireRelease, scope);
        }

        // case seq_cst:
        buildSingleMemOrderCase(builder, caseAttrs, regions, loc,
                                order.getType(),
                                mlir::cir::MemOrder::SequentiallyConsistent);
        buildAtomicOp(*this, e, dest, ptr, val1, val2, isWeak, orderFail, size,
                      mlir::cir::MemOrder::SequentiallyConsistent, scope);

        os.addRegions(regions);
        os.addAttribute("cases", builder.getArrayAttr(caseAttrs));
      });

  if (rValTy->isVoidType())
    return RValue::get(nullptr);
  return convertTempToRValue(dest.withElementType(convertTypeForMem(rValTy)),
                             rValTy, e->getExprLoc());
}

void CIRGenFunction::buildAtomicStore(RValue rvalue, LValue lvalue,
                                      bool isInit) {
  bool isVolatile = lvalue.isVolatileQualified();
  mlir::cir::MemOrder mo;
  if (lvalue.getType()->isAtomicType()) {
    mo = mlir::cir::MemOrder::SequentiallyConsistent;
  } else {
    mo = mlir::cir::MemOrder::Release;
    isVolatile = true;
  }
  return buildAtomicStore(rvalue, lvalue, mo, isVolatile, isInit);
}

/// Return true if \param ValTy is a type that should be casted to integer
/// around the atomic memory operation. If \param CmpXchg is true, then the
/// cast of a floating point type is made as that instruction can not have
/// floating point operands.  TODO: Allow compare-and-exchange and FP - see
/// comment in CIRGenAtomicExpandPass.cpp.
static bool shouldCastToInt(mlir::Type valTy, bool cmpXchg) {
  if (mlir::cir::isAnyFloatingPointType(valTy))
    return isa<mlir::cir::FP80Type>(valTy) || cmpXchg;
  return !isa<mlir::cir::IntType>(valTy) && !isa<mlir::cir::PointerType>(valTy);
}

mlir::Value AtomicInfo::getScalarRValValueOrNull(RValue rVal) const {
  if (rVal.isScalar() && (!hasPadding() || !LVal.isSimple()))
    return rVal.getScalarVal();
  return nullptr;
}

/// Materialize an r-value into memory for the purposes of storing it
/// to an atomic type.
Address AtomicInfo::materializeRValue(RValue rvalue) const {
  // Aggregate r-values are already in memory, and EmitAtomicStore
  // requires them to be values of the atomic type.
  if (rvalue.isAggregate())
    return rvalue.getAggregateAddress();

  // Otherwise, make a temporary and materialize into it.
  LValue tempLv = cgf.makeAddrLValue(createTempAlloca(), getAtomicType());
  AtomicInfo atomics(cgf, tempLv, tempLv.getAddress().getPointer().getLoc());
  atomics.emitCopyIntoMemory(rvalue);
  return tempLv.getAddress();
}

bool AtomicInfo::emitMemSetZeroIfNecessary() const {
  assert(LVal.isSimple());
  Address addr = LVal.getAddress();
  if (!requiresMemSetZero(addr.getElementType()))
    return false;

  llvm_unreachable("NYI");
}

/// Copy an r-value into memory as part of storing to an atomic type.
/// This needs to create a bit-pattern suitable for atomic operations.
void AtomicInfo::emitCopyIntoMemory(RValue rvalue) const {
  assert(LVal.isSimple());
  // If we have an r-value, the rvalue should be of the atomic type,
  // which means that the caller is responsible for having zeroed
  // any padding.  Just do an aggregate copy of that type.
  if (rvalue.isAggregate()) {
    llvm_unreachable("NYI");
    return;
  }

  // Okay, otherwise we're copying stuff.

  // Zero out the buffer if necessary.
  emitMemSetZeroIfNecessary();

  // Drill past the padding if present.
  LValue tempLVal = projectValue();

  // Okay, store the rvalue in.
  if (rvalue.isScalar()) {
    cgf.buildStoreOfScalar(rvalue.getScalarVal(), tempLVal, /*init*/ true);
  } else {
    llvm_unreachable("NYI");
  }
}

mlir::Value AtomicInfo::convertRValueToInt(RValue rVal, bool cmpXchg) const {
  // If we've got a scalar value of the right size, try to avoid going
  // through memory. Floats get casted if needed by AtomicExpandPass.
  if (auto value = getScalarRValValueOrNull(rVal)) {
    if (!shouldCastToInt(value.getType(), cmpXchg)) {
      return cgf.buildToMemory(value, valueTy);
    }
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
}

/// Emit a store to an l-value of atomic type.
///
/// Note that the r-value is expected to be an r-value *of the atomic
/// type*; this means that for aggregate r-values, it should include
/// storage for any padding that was necessary.
void CIRGenFunction::buildAtomicStore(RValue rvalue, LValue dest,
                                      mlir::cir::MemOrder mo, bool isVolatile,
                                      bool isInit) {
  // If this is an aggregate r-value, it should agree in type except
  // maybe for address-space qualification.
  auto loc = dest.getPointer().getLoc();
  assert(!rvalue.isAggregate() ||
         rvalue.getAggregateAddress().getElementType() ==
             dest.getAddress().getElementType());

  AtomicInfo atomics(*this, dest, loc);
  LValue lVal = atomics.getAtomicLValue();

  // If this is an initialization, just put the value there normally.
  if (lVal.isSimple()) {
    if (isInit) {
      atomics.emitCopyIntoMemory(rvalue);
      return;
    }

    // Check whether we should use a library call.
    if (atomics.shouldUseLibcall()) {
      llvm_unreachable("NYI");
    }

    // Okay, we're doing this natively.
    auto valToStore = atomics.convertRValueToInt(rvalue);

    // Do the atomic store.
    Address addr = atomics.getAtomicAddress();
    if (auto value = atomics.getScalarRValValueOrNull(rvalue))
      if (shouldCastToInt(value.getType(), /*CmpXchg=*/false)) {
        addr = atomics.castToAtomicIntPointer(addr);
        valToStore = builder.createIntCast(valToStore, addr.getElementType());
      }
    auto store = builder.createStore(loc, valToStore, addr);

    if (mo == mlir::cir::MemOrder::Acquire)
      mo = mlir::cir::MemOrder::Relaxed; // Monotonic
    else if (mo == mlir::cir::MemOrder::AcquireRelease)
      mo = mlir::cir::MemOrder::Release;
    // Initializations don't need to be atomic.
    if (!isInit)
      store.setMemOrder(mo);

    // Other decoration.
    if (isVolatile)
      store.setIsVolatile(true);

    // DecorateInstructionWithTBAA
    assert(!MissingFeatures::tbaa());
    return;
  }

  llvm_unreachable("NYI");
}

void CIRGenFunction::buildAtomicInit(Expr *init, LValue dest) {
  AtomicInfo atomics(*this, dest, getLoc(init->getSourceRange()));

  switch (atomics.getEvaluationKind()) {
  case TEK_Scalar: {
    mlir::Value value = buildScalarExpr(init);
    atomics.emitCopyIntoMemory(RValue::get(value));
    return;
  }

  case TEK_Complex: {
    llvm_unreachable("NYI");
    return;
  }

  case TEK_Aggregate: {
    // Fix up the destination if the initializer isn't an expression
    // of atomic type.
    llvm_unreachable("NYI");
    return;
  }
  }
  llvm_unreachable("bad evaluation kind");
}
