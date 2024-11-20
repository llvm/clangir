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
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

namespace {
class AtomicInfo {
  CIRGenFunction &CGF;
  QualType AtomicTy;
  QualType ValueTy;
  uint64_t AtomicSizeInBits;
  uint64_t ValueSizeInBits;
  CharUnits AtomicAlign;
  CharUnits ValueAlign;
  TypeEvaluationKind EvaluationKind;
  bool UseLibcall;
  LValue LVal;
  CIRGenBitFieldInfo BFI;
  mlir::Location loc;

public:
  AtomicInfo(CIRGenFunction &CGF, LValue &lvalue, mlir::Location l)
      : CGF(CGF), AtomicSizeInBits(0), ValueSizeInBits(0),
        EvaluationKind(TEK_Scalar), UseLibcall(true), loc(l) {
    assert(!lvalue.isGlobalReg());
    ASTContext &C = CGF.getContext();
    if (lvalue.isSimple()) {
      AtomicTy = lvalue.getType();
      if (auto *ATy = AtomicTy->getAs<AtomicType>())
        ValueTy = ATy->getValueType();
      else
        ValueTy = AtomicTy;
      EvaluationKind = CGF.getEvaluationKind(ValueTy);

      uint64_t ValueAlignInBits;
      uint64_t AtomicAlignInBits;
      TypeInfo ValueTI = C.getTypeInfo(ValueTy);
      ValueSizeInBits = ValueTI.Width;
      ValueAlignInBits = ValueTI.Align;

      TypeInfo AtomicTI = C.getTypeInfo(AtomicTy);
      AtomicSizeInBits = AtomicTI.Width;
      AtomicAlignInBits = AtomicTI.Align;

      assert(ValueSizeInBits <= AtomicSizeInBits);
      assert(ValueAlignInBits <= AtomicAlignInBits);

      AtomicAlign = C.toCharUnitsFromBits(AtomicAlignInBits);
      ValueAlign = C.toCharUnitsFromBits(ValueAlignInBits);
      if (lvalue.getAlignment().isZero())
        lvalue.setAlignment(AtomicAlign);

      LVal = lvalue;
    } else if (lvalue.isBitField()) {
      llvm_unreachable("NYI");
    } else if (lvalue.isVectorElt()) {
      ValueTy = lvalue.getType()->castAs<VectorType>()->getElementType();
      ValueSizeInBits = C.getTypeSize(ValueTy);
      AtomicTy = lvalue.getType();
      AtomicSizeInBits = C.getTypeSize(AtomicTy);
      AtomicAlign = ValueAlign = lvalue.getAlignment();
      LVal = lvalue;
    } else {
      llvm_unreachable("NYI");
    }
    UseLibcall = !C.getTargetInfo().hasBuiltinAtomic(
        AtomicSizeInBits, C.toBits(lvalue.getAlignment()));
  }

  QualType getAtomicType() const { return AtomicTy; }
  QualType getValueType() const { return ValueTy; }
  CharUnits getAtomicAlignment() const { return AtomicAlign; }
  uint64_t getAtomicSizeInBits() const { return AtomicSizeInBits; }
  uint64_t getValueSizeInBits() const { return ValueSizeInBits; }
  TypeEvaluationKind getEvaluationKind() const { return EvaluationKind; }
  bool shouldUseLibcall() const { return UseLibcall; }
  const LValue &getAtomicLValue() const { return LVal; }
  mlir::Value getAtomicPointer() const {
    if (LVal.isSimple())
      return LVal.getPointer();
    else if (LVal.isBitField())
      return LVal.getBitFieldPointer();
    else if (LVal.isVectorElt())
      return LVal.getVectorPointer();
    assert(LVal.isExtVectorElt());
    // TODO(cir): return LVal.getExtVectorPointer();
    llvm_unreachable("NYI");
  }
  Address getAtomicAddress() const {
    mlir::Type ElTy;
    if (LVal.isSimple())
      ElTy = LVal.getAddress().getElementType();
    else if (LVal.isBitField())
      ElTy = LVal.getBitFieldAddress().getElementType();
    else if (LVal.isVectorElt())
      ElTy = LVal.getVectorAddress().getElementType();
    else // TODO(cir): ElTy = LVal.getExtVectorAddress().getElementType();
      llvm_unreachable("NYI");
    return Address(getAtomicPointer(), ElTy, getAtomicAlignment());
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

  mlir::Value getScalarRValValueOrNull(RValue RVal) const;

  /// Cast the given pointer to an integer pointer suitable for atomic
  /// operations if the source.
  Address castToAtomicIntPointer(Address Addr) const;

  /// If Addr is compatible with the iN that will be used for an atomic
  /// operation, bitcast it. Otherwise, create a temporary that is suitable
  /// and copy the value across.
  Address convertToAtomicIntPointer(Address Addr) const;

  /// Turn an atomic-layout object into an r-value.
  RValue convertAtomicTempToRValue(Address addr, AggValueSlot resultSlot,
                                   SourceLocation loc, bool AsValue) const;

  /// Converts a rvalue to integer value.
  mlir::Value convertRValueToInt(RValue RVal, bool CmpXchg = false) const;

  RValue ConvertIntToValueOrAtomic(mlir::Value IntVal, AggValueSlot ResultSlot,
                                   SourceLocation Loc, bool AsValue) const;

  /// Copy an atomic r-value into atomic-layout memory.
  void emitCopyIntoMemory(RValue rvalue) const;

  /// Project an l-value down to the value field.
  LValue projectValue() const {
    assert(LVal.isSimple());
    Address addr = getAtomicAddress();
    if (hasPadding())
      llvm_unreachable("NYI");

    return LValue::makeAddr(addr, getValueType(), CGF.getContext(),
                            LVal.getBaseInfo());
  }

  /// Emits atomic load.
  /// \returns Loaded value.
  RValue EmitAtomicLoad(AggValueSlot ResultSlot, SourceLocation Loc,
                        bool AsValue, llvm::AtomicOrdering AO, bool IsVolatile);

  /// Emits atomic compare-and-exchange sequence.
  /// \param Expected Expected value.
  /// \param Desired Desired value.
  /// \param Success Atomic ordering for success operation.
  /// \param Failure Atomic ordering for failed operation.
  /// \param IsWeak true if atomic operation is weak, false otherwise.
  /// \returns Pair of values: previous value from storage (value type) and
  /// boolean flag (i1 type) with true if success and false otherwise.
  std::pair<RValue, mlir::Value>
  EmitAtomicCompareExchange(RValue Expected, RValue Desired,
                            llvm::AtomicOrdering Success =
                                llvm::AtomicOrdering::SequentiallyConsistent,
                            llvm::AtomicOrdering Failure =
                                llvm::AtomicOrdering::SequentiallyConsistent,
                            bool IsWeak = false);

  /// Emits atomic update.
  /// \param AO Atomic ordering.
  /// \param UpdateOp Update operation for the current lvalue.
  void EmitAtomicUpdate(llvm::AtomicOrdering AO,
                        const llvm::function_ref<RValue(RValue)> &UpdateOp,
                        bool IsVolatile);
  /// Emits atomic update.
  /// \param AO Atomic ordering.
  void EmitAtomicUpdate(llvm::AtomicOrdering AO, RValue UpdateRVal,
                        bool IsVolatile);

  /// Materialize an atomic r-value in atomic-layout memory.
  Address materializeRValue(RValue rvalue) const;

  /// Creates temp alloca for intermediate operations on atomic value.
  Address CreateTempAlloca() const;

private:
  bool requiresMemSetZero(mlir::Type ty) const;

  /// Emits atomic load as a libcall.
  void EmitAtomicLoadLibcall(mlir::Value AddForLoaded, llvm::AtomicOrdering AO,
                             bool IsVolatile);
  /// Emits atomic load as LLVM instruction.
  mlir::Value EmitAtomicLoadOp(llvm::AtomicOrdering AO, bool IsVolatile);
  /// Emits atomic compare-and-exchange op as a libcall.
  mlir::Value EmitAtomicCompareExchangeLibcall(
      mlir::Value ExpectedAddr, mlir::Value DesiredAddr,
      llvm::AtomicOrdering Success =
          llvm::AtomicOrdering::SequentiallyConsistent,
      llvm::AtomicOrdering Failure =
          llvm::AtomicOrdering::SequentiallyConsistent);
  /// Emits atomic compare-and-exchange op as LLVM instruction.
  std::pair<mlir::Value, mlir::Value>
  EmitAtomicCompareExchangeOp(mlir::Value ExpectedVal, mlir::Value DesiredVal,
                              llvm::AtomicOrdering Success =
                                  llvm::AtomicOrdering::SequentiallyConsistent,
                              llvm::AtomicOrdering Failure =
                                  llvm::AtomicOrdering::SequentiallyConsistent,
                              bool IsWeak = false);
  /// Emit atomic update as libcalls.
  void
  EmitAtomicUpdateLibcall(llvm::AtomicOrdering AO,
                          const llvm::function_ref<RValue(RValue)> &UpdateOp,
                          bool IsVolatile);
  /// Emit atomic update as LLVM instructions.
  void EmitAtomicUpdateOp(llvm::AtomicOrdering AO,
                          const llvm::function_ref<RValue(RValue)> &UpdateOp,
                          bool IsVolatile);
  /// Emit atomic update as libcalls.
  void EmitAtomicUpdateLibcall(llvm::AtomicOrdering AO, RValue UpdateRVal,
                               bool IsVolatile);
  /// Emit atomic update as LLVM instructions.
  void EmitAtomicUpdateOp(llvm::AtomicOrdering AO, RValue UpdateRal,
                          bool IsVolatile);
};
} // namespace

// This function emits any expression (scalar, complex, or aggregate)
// into a temporary alloca.
static Address buildValToTemp(CIRGenFunction &CGF, Expr *E) {
  Address DeclPtr = CGF.CreateMemTemp(
      E->getType(), CGF.getLoc(E->getSourceRange()), ".atomictmp");
  CGF.buildAnyExprToMem(E, DeclPtr, E->getType().getQualifiers(),
                        /*Init*/ true);
  return DeclPtr;
}

/// Does a store of the given IR type modify the full expected width?
static bool isFullSizeType(CIRGenModule &CGM, mlir::Type ty,
                           uint64_t expectedSize) {
  return (CGM.getDataLayout().getTypeStoreSize(ty) * 8 == expectedSize);
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
    return !isFullSizeType(CGF.CGM, ty, AtomicSizeInBits);
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
  auto ty = CGF.getBuilder().getUIntNTy(AtomicSizeInBits);
  return addr.withElementType(ty);
}

Address AtomicInfo::convertToAtomicIntPointer(Address Addr) const {
  auto Ty = Addr.getElementType();
  uint64_t SourceSizeInBits = CGF.CGM.getDataLayout().getTypeSizeInBits(Ty);
  if (SourceSizeInBits != AtomicSizeInBits) {
    llvm_unreachable("NYI");
  }

  return castToAtomicIntPointer(Addr);
}

Address AtomicInfo::CreateTempAlloca() const {
  Address TempAlloca = CGF.CreateMemTemp(
      (LVal.isBitField() && ValueSizeInBits > AtomicSizeInBits) ? ValueTy
                                                                : AtomicTy,
      getAtomicAlignment(), loc, "atomic-temp");
  // Cast to pointer to value type for bitfields.
  if (LVal.isBitField()) {
    llvm_unreachable("NYI");
  }
  return TempAlloca;
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
    } else if (mlir::isa<mlir::cir::BoolType>(c.getType())) {
      val = mlir::cast<mlir::cir::BoolAttr>(c.getValue()).getValue();
      return true;
    }
  }
  return false;
}

static void buildAtomicCmpXchg(CIRGenFunction &CGF, AtomicExpr *E, bool IsWeak,
                               Address Dest, Address Ptr, Address Val1,
                               Address Val2, uint64_t Size,
                               mlir::cir::MemOrder SuccessOrder,
                               mlir::cir::MemOrder FailureOrder,
                               llvm::SyncScope::ID Scope) {
  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(E->getSourceRange());
  auto Expected = builder.createLoad(loc, Val1);
  auto Desired = builder.createLoad(loc, Val2);
  auto boolTy = builder.getBoolTy();
  auto cmpxchg = builder.create<mlir::cir::AtomicCmpXchg>(
      loc, Expected.getType(), boolTy, Ptr.getPointer(), Expected, Desired,
      SuccessOrder, FailureOrder);
  cmpxchg.setIsVolatile(E->isVolatile());
  cmpxchg.setWeak(IsWeak);

  auto cmp = builder.createNot(cmpxchg.getCmp());
  builder.create<mlir::cir::IfOp>(
      loc, cmp, false, [&](mlir::OpBuilder &, mlir::Location) {
        auto ptrTy =
            mlir::cast<mlir::cir::PointerType>(Val1.getPointer().getType());
        if (Val1.getElementType() != ptrTy.getPointee()) {
          Val1 = Val1.withPointer(builder.createPtrBitcast(
              Val1.getPointer(), Val1.getElementType()));
        }
        builder.createStore(loc, cmpxchg.getOld(), Val1);
        builder.createYield(loc);
      });

  // Update the memory at Dest with Cmp's value.
  CGF.buildStoreOfScalar(cmpxchg.getCmp(),
                         CGF.makeAddrLValue(Dest, E->getType()));
}

/// Given an ordering required on success, emit all possible cmpxchg
/// instructions to cope with the provided (but possibly only dynamically known)
/// FailureOrder.
static void buildAtomicCmpXchgFailureSet(
    CIRGenFunction &CGF, AtomicExpr *E, bool IsWeak, Address Dest, Address Ptr,
    Address Val1, Address Val2, mlir::Value FailureOrderVal, uint64_t Size,
    mlir::cir::MemOrder SuccessOrder, llvm::SyncScope::ID Scope) {

  mlir::cir::MemOrder FailureOrder;
  if (auto ordAttr = getConstOpIntAttr(FailureOrderVal)) {
    // We should not ever get to a case where the ordering isn't a valid CABI
    // value, but it's hard to enforce that in general.
    auto ord = ordAttr.getUInt();
    if (!mlir::cir::isValidCIRAtomicOrderingCABI(ord)) {
      FailureOrder = mlir::cir::MemOrder::Relaxed;
    } else {
      switch ((mlir::cir::MemOrder)ord) {
      case mlir::cir::MemOrder::Relaxed:
        // 31.7.2.18: "The failure argument shall not be memory_order_release
        // nor memory_order_acq_rel". Fallback to monotonic.
      case mlir::cir::MemOrder::Release:
      case mlir::cir::MemOrder::AcquireRelease:
        FailureOrder = mlir::cir::MemOrder::Relaxed;
        break;
      case mlir::cir::MemOrder::Consume:
      case mlir::cir::MemOrder::Acquire:
        FailureOrder = mlir::cir::MemOrder::Acquire;
        break;
      case mlir::cir::MemOrder::SequentiallyConsistent:
        FailureOrder = mlir::cir::MemOrder::SequentiallyConsistent;
        break;
      }
    }
    // Prior to c++17, "the failure argument shall be no stronger than the
    // success argument". This condition has been lifted and the only
    // precondition is 31.7.2.18. Effectively treat this as a DR and skip
    // language version checks.
    buildAtomicCmpXchg(CGF, E, IsWeak, Dest, Ptr, Val1, Val2, Size,
                       SuccessOrder, FailureOrder, Scope);
    return;
  }

  llvm_unreachable("NYI");
}

static void buildAtomicOp(CIRGenFunction &CGF, AtomicExpr *E, Address Dest,
                          Address Ptr, Address Val1, Address Val2,
                          mlir::Value IsWeak, mlir::Value FailureOrder,
                          uint64_t Size, mlir::cir::MemOrder Order,
                          uint8_t Scope) {
  assert(!MissingFeatures::syncScopeID());
  StringRef Op;

  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(E->getSourceRange());
  auto orderAttr = mlir::cir::MemOrderAttr::get(builder.getContext(), Order);
  mlir::cir::AtomicFetchKindAttr fetchAttr;
  bool fetchFirst = true;

  switch (E->getOp()) {
  case AtomicExpr::AO__c11_atomic_init:
  case AtomicExpr::AO__opencl_atomic_init:
    llvm_unreachable("Already handled!");

  case AtomicExpr::AO__c11_atomic_compare_exchange_strong:
  case AtomicExpr::AO__hip_atomic_compare_exchange_strong:
  case AtomicExpr::AO__opencl_atomic_compare_exchange_strong:
    buildAtomicCmpXchgFailureSet(CGF, E, false, Dest, Ptr, Val1, Val2,
                                 FailureOrder, Size, Order, Scope);
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
    if (isCstWeak(IsWeak, weakVal)) {
      buildAtomicCmpXchgFailureSet(CGF, E, weakVal, Dest, Ptr, Val1, Val2,
                                   FailureOrder, Size, Order, Scope);
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
    auto *load = builder.createLoad(loc, Ptr).getDefiningOp();
    // FIXME(cir): add scope information.
    assert(!MissingFeatures::syncScopeID());
    load->setAttr("mem_order", orderAttr);
    if (E->isVolatile())
      load->setAttr("is_volatile", mlir::UnitAttr::get(builder.getContext()));

    // TODO(cir): this logic should be part of createStore, but doing so
    // currently breaks CodeGen/union.cpp and CodeGen/union.cpp.
    auto ptrTy =
        mlir::cast<mlir::cir::PointerType>(Dest.getPointer().getType());
    if (Dest.getElementType() != ptrTy.getPointee()) {
      Dest = Dest.withPointer(
          builder.createPtrBitcast(Dest.getPointer(), Dest.getElementType()));
    }
    builder.createStore(loc, load->getResult(0), Dest);
    return;
  }

  case AtomicExpr::AO__c11_atomic_store:
  case AtomicExpr::AO__opencl_atomic_store:
  case AtomicExpr::AO__hip_atomic_store:
  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__atomic_store_n:
  case AtomicExpr::AO__scoped_atomic_store:
  case AtomicExpr::AO__scoped_atomic_store_n: {
    auto loadVal1 = builder.createLoad(loc, Val1);
    // FIXME(cir): add scope information.
    assert(!MissingFeatures::syncScopeID());
    builder.createStore(loc, loadVal1, Ptr, E->isVolatile(),
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

  assert(Op.size() && "expected operation name to build");
  auto LoadVal1 = builder.createLoad(loc, Val1);

  SmallVector<mlir::Value> atomicOperands = {Ptr.getPointer(), LoadVal1};
  SmallVector<mlir::Type> atomicResTys = {LoadVal1.getType()};
  auto RMWI = builder.create(loc, builder.getStringAttr(Op), atomicOperands,
                             atomicResTys, {});

  if (fetchAttr)
    RMWI->setAttr("binop", fetchAttr);
  RMWI->setAttr("mem_order", orderAttr);
  if (E->isVolatile())
    RMWI->setAttr("is_volatile", mlir::UnitAttr::get(builder.getContext()));
  if (fetchFirst && Op == mlir::cir::AtomicFetch::getOperationName())
    RMWI->setAttr("fetch_first", mlir::UnitAttr::get(builder.getContext()));

  auto Result = RMWI->getResult(0);

  // TODO(cir): this logic should be part of createStore, but doing so currently
  // breaks CodeGen/union.cpp and CodeGen/union.cpp.
  auto ptrTy = mlir::cast<mlir::cir::PointerType>(Dest.getPointer().getType());
  if (Dest.getElementType() != ptrTy.getPointee()) {
    Dest = Dest.withPointer(
        builder.createPtrBitcast(Dest.getPointer(), Dest.getElementType()));
  }
  builder.createStore(loc, Result, Dest);
}

static RValue buildAtomicLibcall(CIRGenFunction &CGF, StringRef fnName,
                                 QualType resultType, CallArgList &args) {
  [[maybe_unused]] const CIRGenFunctionInfo &fnInfo =
      CGF.CGM.getTypes().arrangeBuiltinFunctionCall(resultType, args);
  [[maybe_unused]] auto fnTy = CGF.CGM.getTypes().GetFunctionType(fnInfo);
  llvm_unreachable("NYI");
}

static void buildAtomicOp(CIRGenFunction &CGF, AtomicExpr *Expr, Address Dest,
                          Address Ptr, Address Val1, Address Val2,
                          mlir::Value IsWeak, mlir::Value FailureOrder,
                          uint64_t Size, mlir::cir::MemOrder Order,
                          mlir::Value Scope) {
  auto ScopeModel = Expr->getScopeModel();

  // LLVM atomic instructions always have synch scope. If clang atomic
  // expression has no scope operand, use default LLVM synch scope.
  if (!ScopeModel) {
    assert(!MissingFeatures::syncScopeID());
    buildAtomicOp(CGF, Expr, Dest, Ptr, Val1, Val2, IsWeak, FailureOrder, Size,
                  Order, /*FIXME(cir): LLVM default scope*/ 1);
    return;
  }

  // Handle constant scope.
  if (getConstOpIntAttr(Scope)) {
    assert(!MissingFeatures::syncScopeID());
    llvm_unreachable("NYI");
    return;
  }

  // Handle non-constant scope.
  llvm_unreachable("NYI");
}

RValue CIRGenFunction::buildAtomicExpr(AtomicExpr *E) {
  QualType AtomicTy = E->getPtr()->getType()->getPointeeType();
  QualType MemTy = AtomicTy;
  if (const AtomicType *AT = AtomicTy->getAs<AtomicType>())
    MemTy = AT->getValueType();
  mlir::Value IsWeak = nullptr, OrderFail = nullptr;

  Address Val1 = Address::invalid();
  Address Val2 = Address::invalid();
  Address Dest = Address::invalid();
  Address Ptr = buildPointerWithAlignment(E->getPtr());

  if (E->getOp() == AtomicExpr::AO__c11_atomic_init ||
      E->getOp() == AtomicExpr::AO__opencl_atomic_init) {
    LValue lvalue = makeAddrLValue(Ptr, AtomicTy);
    buildAtomicInit(E->getVal1(), lvalue);
    return RValue::get(nullptr);
  }

  auto TInfo = getContext().getTypeInfoInChars(AtomicTy);
  uint64_t Size = TInfo.Width.getQuantity();
  unsigned MaxInlineWidthInBits = getTarget().getMaxAtomicInlineWidth();

  CharUnits MaxInlineWidth =
      getContext().toCharUnitsFromBits(MaxInlineWidthInBits);
  DiagnosticsEngine &Diags = CGM.getDiags();
  bool Misaligned = (Ptr.getAlignment() % TInfo.Width) != 0;
  bool Oversized = getContext().toBits(TInfo.Width) > MaxInlineWidthInBits;
  if (Misaligned) {
    Diags.Report(E->getBeginLoc(), diag::warn_atomic_op_misaligned)
        << (int)TInfo.Width.getQuantity()
        << (int)Ptr.getAlignment().getQuantity();
  }
  if (Oversized) {
    Diags.Report(E->getBeginLoc(), diag::warn_atomic_op_oversized)
        << (int)TInfo.Width.getQuantity() << (int)MaxInlineWidth.getQuantity();
  }

  auto Order = buildScalarExpr(E->getOrder());
  auto Scope = E->getScopeModel() ? buildScalarExpr(E->getScope()) : nullptr;
  bool ShouldCastToIntPtrTy = true;

  switch (E->getOp()) {
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
    Dest = buildPointerWithAlignment(E->getVal1());
    break;

  case AtomicExpr::AO__atomic_store:
  case AtomicExpr::AO__scoped_atomic_store:
    Val1 = buildPointerWithAlignment(E->getVal1());
    break;

  case AtomicExpr::AO__atomic_exchange:
  case AtomicExpr::AO__scoped_atomic_exchange:
    Val1 = buildPointerWithAlignment(E->getVal1());
    Dest = buildPointerWithAlignment(E->getVal2());
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
    Val1 = buildPointerWithAlignment(E->getVal1());
    if (E->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
        E->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange)
      Val2 = buildPointerWithAlignment(E->getVal2());
    else
      Val2 = buildValToTemp(*this, E->getVal2());
    OrderFail = buildScalarExpr(E->getOrderFail());
    if (E->getOp() == AtomicExpr::AO__atomic_compare_exchange_n ||
        E->getOp() == AtomicExpr::AO__atomic_compare_exchange ||
        E->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange_n ||
        E->getOp() == AtomicExpr::AO__scoped_atomic_compare_exchange) {
      IsWeak = buildScalarExpr(E->getWeak());
    }
    break;

  case AtomicExpr::AO__c11_atomic_fetch_add:
  case AtomicExpr::AO__c11_atomic_fetch_sub:
  case AtomicExpr::AO__hip_atomic_fetch_add:
  case AtomicExpr::AO__hip_atomic_fetch_sub:
  case AtomicExpr::AO__opencl_atomic_fetch_add:
  case AtomicExpr::AO__opencl_atomic_fetch_sub:
    if (MemTy->isPointerType()) {
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
    ShouldCastToIntPtrTy = !MemTy->isFloatingType();
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
    Val1 = buildValToTemp(*this, E->getVal1());
    break;
  }

  QualType RValTy = E->getType().getUnqualifiedType();

  // The inlined atomics only function on iN types, where N is a power of 2. We
  // need to make sure (via temporaries if necessary) that all incoming values
  // are compatible.
  LValue AtomicVal = makeAddrLValue(Ptr, AtomicTy);
  AtomicInfo Atomics(*this, AtomicVal, getLoc(E->getSourceRange()));

  if (ShouldCastToIntPtrTy) {
    Ptr = Atomics.castToAtomicIntPointer(Ptr);
    if (Val1.isValid())
      Val1 = Atomics.convertToAtomicIntPointer(Val1);
    if (Val2.isValid())
      Val2 = Atomics.convertToAtomicIntPointer(Val2);
  }
  if (Dest.isValid()) {
    if (ShouldCastToIntPtrTy)
      Dest = Atomics.castToAtomicIntPointer(Dest);
  } else if (E->isCmpXChg())
    Dest = CreateMemTemp(RValTy, getLoc(E->getSourceRange()), "cmpxchg.bool");
  else if (!RValTy->isVoidType()) {
    Dest = Atomics.CreateTempAlloca();
    if (ShouldCastToIntPtrTy)
      Dest = Atomics.castToAtomicIntPointer(Dest);
  }

  bool PowerOf2Size = (Size & (Size - 1)) == 0;
  bool UseLibcall = !PowerOf2Size || (Size > 16);

  // For atomics larger than 16 bytes, emit a libcall from the frontend. This
  // avoids the overhead of dealing with excessively-large value types in IR.
  // Non-power-of-2 values also lower to libcall here, as they are not currently
  // permitted in IR instructions (although that constraint could be relaxed in
  // the future). For other cases where a libcall is required on a given
  // platform, we let the backend handle it (this includes handling for all of
  // the size-optimized libcall variants, which are only valid up to 16 bytes.)
  //
  // See: https://llvm.org/docs/Atomics.html#libcalls-atomic
  if (UseLibcall) {
    CallArgList Args;
    // For non-optimized library calls, the size is the first parameter.
    Args.add(RValue::get(builder.getConstInt(getLoc(E->getSourceRange()),
                                             SizeTy, Size)),
             getContext().getSizeType());

    // The atomic address is the second parameter.
    // The OpenCL atomic library functions only accept pointer arguments to
    // generic address space.
    auto CastToGenericAddrSpace = [&](mlir::Value V, QualType PT) {
      if (!E->isOpenCL())
        return V;
      llvm_unreachable("NYI");
    };

    Args.add(RValue::get(CastToGenericAddrSpace(Ptr.emitRawPointer(),
                                                E->getPtr()->getType())),
             getContext().VoidPtrTy);

    // The next 1-3 parameters are op-dependent.
    std::string LibCallName;
    QualType RetTy;
    bool HaveRetTy = false;
    switch (E->getOp()) {
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
      LibCallName = "__atomic_compare_exchange";
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
      LibCallName = "__atomic_exchange";
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
      LibCallName = "__atomic_store";
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
      LibCallName = "__atomic_load";
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

    if (E->isOpenCL()) {
      LibCallName =
          std::string("__opencl") + StringRef(LibCallName).drop_front(1).str();
    }
    // By default, assume we return a value of the atomic type.
    if (!HaveRetTy) {
      llvm_unreachable("NYI");
    }
    // Order is always the last parameter.
    Args.add(RValue::get(Order), getContext().IntTy);
    if (E->isOpenCL()) {
      llvm_unreachable("NYI");
    }

    [[maybe_unused]] RValue Res =
        buildAtomicLibcall(*this, LibCallName, RetTy, Args);
    // The value is returned directly from the libcall.
    if (E->isCmpXChg()) {
      llvm_unreachable("NYI");
    }

    if (RValTy->isVoidType()) {
      llvm_unreachable("NYI");
    }

    llvm_unreachable("NYI");
  }

  [[maybe_unused]] bool IsStore =
      E->getOp() == AtomicExpr::AO__c11_atomic_store ||
      E->getOp() == AtomicExpr::AO__opencl_atomic_store ||
      E->getOp() == AtomicExpr::AO__hip_atomic_store ||
      E->getOp() == AtomicExpr::AO__atomic_store ||
      E->getOp() == AtomicExpr::AO__atomic_store_n ||
      E->getOp() == AtomicExpr::AO__scoped_atomic_store ||
      E->getOp() == AtomicExpr::AO__scoped_atomic_store_n;
  [[maybe_unused]] bool IsLoad =
      E->getOp() == AtomicExpr::AO__c11_atomic_load ||
      E->getOp() == AtomicExpr::AO__opencl_atomic_load ||
      E->getOp() == AtomicExpr::AO__hip_atomic_load ||
      E->getOp() == AtomicExpr::AO__atomic_load ||
      E->getOp() == AtomicExpr::AO__atomic_load_n ||
      E->getOp() == AtomicExpr::AO__scoped_atomic_load ||
      E->getOp() == AtomicExpr::AO__scoped_atomic_load_n;

  if (auto ordAttr = getConstOpIntAttr(Order)) {
    // We should not ever get to a case where the ordering isn't a valid CABI
    // value, but it's hard to enforce that in general.
    auto ord = ordAttr.getUInt();
    if (mlir::cir::isValidCIRAtomicOrderingCABI(ord)) {
      switch ((mlir::cir::MemOrder)ord) {
      case mlir::cir::MemOrder::Relaxed:
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::Relaxed, Scope);
        break;
      case mlir::cir::MemOrder::Consume:
      case mlir::cir::MemOrder::Acquire:
        if (IsStore)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::Acquire, Scope);
        break;
      case mlir::cir::MemOrder::Release:
        if (IsLoad)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::Release, Scope);
        break;
      case mlir::cir::MemOrder::AcquireRelease:
        if (IsLoad || IsStore)
          break; // Avoid crashing on code with undefined behavior
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::AcquireRelease, Scope);
        break;
      case mlir::cir::MemOrder::SequentiallyConsistent:
        buildAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                      mlir::cir::MemOrder::SequentiallyConsistent, Scope);
        break;
      }
    }
    if (RValTy->isVoidType())
      return RValue::get(nullptr);

    return convertTempToRValue(Dest.withElementType(convertTypeForMem(RValTy)),
                               RValTy, E->getExprLoc());
  }

  // Long case, when Order isn't obviously constant.
  llvm_unreachable("NYI");
}

void CIRGenFunction::buildAtomicStore(RValue rvalue, LValue lvalue,
                                      bool isInit) {
  bool IsVolatile = lvalue.isVolatileQualified();
  mlir::cir::MemOrder MO;
  if (lvalue.getType()->isAtomicType()) {
    MO = mlir::cir::MemOrder::SequentiallyConsistent;
  } else {
    MO = mlir::cir::MemOrder::Release;
    IsVolatile = true;
  }
  return buildAtomicStore(rvalue, lvalue, MO, IsVolatile, isInit);
}

/// Return true if \param ValTy is a type that should be casted to integer
/// around the atomic memory operation. If \param CmpXchg is true, then the
/// cast of a floating point type is made as that instruction can not have
/// floating point operands.  TODO: Allow compare-and-exchange and FP - see
/// comment in CIRGenAtomicExpandPass.cpp.
static bool shouldCastToInt(mlir::Type ValTy, bool CmpXchg) {
  if (mlir::cir::isAnyFloatingPointType(ValTy))
    return isa<mlir::cir::FP80Type>(ValTy) || CmpXchg;
  return !isa<mlir::cir::IntType>(ValTy) && !isa<mlir::cir::PointerType>(ValTy);
}

mlir::Value AtomicInfo::getScalarRValValueOrNull(RValue RVal) const {
  if (RVal.isScalar() && (!hasPadding() || !LVal.isSimple()))
    return RVal.getScalarVal();
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
  LValue TempLV = CGF.makeAddrLValue(CreateTempAlloca(), getAtomicType());
  AtomicInfo Atomics(CGF, TempLV, TempLV.getAddress().getPointer().getLoc());
  Atomics.emitCopyIntoMemory(rvalue);
  return TempLV.getAddress();
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
  LValue TempLVal = projectValue();

  // Okay, store the rvalue in.
  if (rvalue.isScalar()) {
    CGF.buildStoreOfScalar(rvalue.getScalarVal(), TempLVal, /*init*/ true);
  } else {
    llvm_unreachable("NYI");
  }
}

mlir::Value AtomicInfo::convertRValueToInt(RValue RVal, bool CmpXchg) const {
  // If we've got a scalar value of the right size, try to avoid going
  // through memory. Floats get casted if needed by AtomicExpandPass.
  if (auto Value = getScalarRValValueOrNull(RVal)) {
    if (!shouldCastToInt(Value.getType(), CmpXchg)) {
      return CGF.buildToMemory(Value, ValueTy);
    } else {
      llvm_unreachable("NYI");
    }
  }

  llvm_unreachable("NYI");
}

/// Emit a store to an l-value of atomic type.
///
/// Note that the r-value is expected to be an r-value *of the atomic
/// type*; this means that for aggregate r-values, it should include
/// storage for any padding that was necessary.
void CIRGenFunction::buildAtomicStore(RValue rvalue, LValue dest,
                                      mlir::cir::MemOrder MO, bool IsVolatile,
                                      bool isInit) {
  // If this is an aggregate r-value, it should agree in type except
  // maybe for address-space qualification.
  auto loc = dest.getPointer().getLoc();
  assert(!rvalue.isAggregate() ||
         rvalue.getAggregateAddress().getElementType() ==
             dest.getAddress().getElementType());

  AtomicInfo atomics(*this, dest, loc);
  LValue LVal = atomics.getAtomicLValue();

  // If this is an initialization, just put the value there normally.
  if (LVal.isSimple()) {
    if (isInit) {
      atomics.emitCopyIntoMemory(rvalue);
      return;
    }

    // Check whether we should use a library call.
    if (atomics.shouldUseLibcall()) {
      llvm_unreachable("NYI");
    }

    // Okay, we're doing this natively.
    auto ValToStore = atomics.convertRValueToInt(rvalue);

    // Do the atomic store.
    Address Addr = atomics.getAtomicAddress();
    if (auto Value = atomics.getScalarRValValueOrNull(rvalue))
      if (shouldCastToInt(Value.getType(), /*CmpXchg=*/false)) {
        Addr = atomics.castToAtomicIntPointer(Addr);
        ValToStore = builder.createIntCast(ValToStore, Addr.getElementType());
      }
    auto store = builder.createStore(loc, ValToStore, Addr);

    if (MO == mlir::cir::MemOrder::Acquire)
      MO = mlir::cir::MemOrder::Relaxed; // Monotonic
    else if (MO == mlir::cir::MemOrder::AcquireRelease)
      MO = mlir::cir::MemOrder::Release;
    // Initializations don't need to be atomic.
    if (!isInit)
      store.setMemOrder(MO);

    // Other decoration.
    if (IsVolatile)
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
