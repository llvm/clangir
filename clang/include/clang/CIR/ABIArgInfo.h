//==-- ABIArgInfo.h - Abstract info regarding ABI-specific arguments -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines ABIArgInfo and associated types used by CIR to track information
// regarding ABI-coerced types for function arguments and return values. This
// was moved to the common library as it might be used by both CIRGen and
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIR_COMMON_ABIARGINFO_H
#define CIR_COMMON_ABIARGINFO_H

#include "mlir/IR/Types.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include <cstdint>

namespace cir {

/// Helper class to encapsulate information about how a specific C
/// type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Pass the argument directly using the normal converted CIR type,
    /// or by coercing to another specified type stored in 'CoerceToType'). If
    /// an offset is specified (in UIntData), then the argument passed is offset
    /// by some number of bytes in the memory representation. A dummy argument
    /// is emitted before the real argument if the specified type stored in
    /// "PaddingType" is not zero.
    Direct,

    /// Valid only for integer argument types. Same as 'direct' but
    /// also emit a zer/sign extension attribute.
    Extend,

    /// Pass the argument indirectly via a hidden pointer with the
    /// specified alignment (0 indicates default alignment) and address space.
    Indirect,

    /// Similar to Indirect, but the pointer may be to an
    /// object that is otherwise referenced. The object is known to not be
    /// modified through any other references for the duration of the call, and
    /// the callee must not itself modify the object. Because C allows parameter
    /// variables to be modified and guarantees that they have unique addresses,
    /// the callee must defensively copy the object into a local variable if it
    /// might be modified or its address might be compared. Since those are
    /// uncommon, in principle this convention allows programs to avoid copies
    /// in more situations. However, it may introduce *extra* copies if the
    /// callee fails to prove that a copy is unnecessary and the caller
    /// naturally produces an unaliased object for the argument.
    IndirectAliased,

    /// Ignore the argument (treat as void). Useful for void and empty
    /// structs.
    Ignore,

    /// Only valid for aggregate argument types. The structure should
    /// be expanded into consecutive arguments for its constituent fields.
    /// Currently expand is only allowed on structures whose fields are all
    /// scalar types or are themselves expandable types.
    Expand,

    /// Only valid for aggregate argument types. The structure
    /// should be expanded into consecutive arguments corresponding to the
    /// non-array elements of the type stored in CoerceToType.
    /// Array elements in the type are assumed to be padding and skipped.
    CoerceAndExpand,

    // TODO: translate this idea to CIR! Define it for now just to ensure that
    // we can assert it not being used
    InAlloca,
    KindFirst = Direct,
    KindLast = InAlloca
  };

private:
  mlir::Type typeData; // canHaveCoerceToType();
  union {
    mlir::Type paddingType;                 // canHavePaddingType()
    mlir::Type unpaddedCoerceAndExpandType; // isCoerceAndExpand()
  };
  struct DirectAttrInfo {
    unsigned offset;
    unsigned align;
  };
  struct IndirectAttrInfo {
    unsigned align;
    unsigned addrSpace;
  };
  union {
    DirectAttrInfo directAttr;     // isDirect() || isExtend()
    IndirectAttrInfo indirectAttr; // isIndirect()
    unsigned allocaFieldIndex;     // isInAlloca()
  };
  Kind theKind;
  bool inReg : 1;          // isDirect() || isExtend() || isIndirect()
  bool canBeFlattened : 1; // isDirect()
  bool signExt : 1;        // isExtend()

  bool canHavePaddingType() const {
    return isDirect() || isExtend() || isIndirect() || isIndirectAliased() ||
           isExpand();
  }

  void setPaddingType(mlir::Type t) {
    assert(canHavePaddingType());
    paddingType = t;
  }

public:
  ABIArgInfo(Kind k = Direct)
      : typeData(nullptr), paddingType(nullptr), directAttr{0, 0}, theKind(k),
        inReg(false), canBeFlattened(false), signExt(false) {}

  static ABIArgInfo getDirect(mlir::Type t = nullptr, unsigned offset = 0,
                              mlir::Type padding = nullptr,
                              bool canBeFlattened = true, unsigned align = 0) {
    auto ai = ABIArgInfo(Direct);
    ai.setCoerceToType(t);
    ai.setPaddingType(padding);
    ai.setDirectOffset(offset);
    ai.setDirectAlign(align);
    ai.setCanBeFlattened(canBeFlattened);
    return ai;
  }

  static ABIArgInfo getSignExtend(clang::QualType ty, mlir::Type t = nullptr) {
    assert(ty->isIntegralOrEnumerationType() && "Unexpected QualType");
    auto ai = ABIArgInfo(Extend);
    ai.setCoerceToType(t);
    ai.setPaddingType(nullptr);
    ai.setDirectOffset(0);
    ai.setDirectAlign(0);
    ai.setSignExt(true);
    return ai;
  }
  static ABIArgInfo getSignExtend(mlir::Type ty, mlir::Type t = nullptr) {
    // NOTE(cir): Enumerations are IntTypes in CIR.
    auto ai = ABIArgInfo(Extend);
    ai.setCoerceToType(t);
    ai.setPaddingType(nullptr);
    ai.setDirectOffset(0);
    ai.setDirectAlign(0);
    ai.setSignExt(true);
    return ai;
  }

  static ABIArgInfo getZeroExtend(clang::QualType ty, mlir::Type t = nullptr) {
    assert(ty->isIntegralOrEnumerationType() && "Unexpected QualType");
    auto ai = ABIArgInfo(Extend);
    ai.setCoerceToType(t);
    ai.setPaddingType(nullptr);
    ai.setDirectOffset(0);
    ai.setDirectAlign(0);
    ai.setSignExt(false);
    return ai;
  }
  static ABIArgInfo getZeroExtend(mlir::Type ty, mlir::Type t = nullptr) {
    // NOTE(cir): Enumerations are IntTypes in CIR.
    assert(mlir::isa<mlir::cir::IntType>(ty) ||
           mlir::isa<mlir::cir::BoolType>(ty));
    auto ai = ABIArgInfo(Extend);
    ai.setCoerceToType(t);
    ai.setPaddingType(nullptr);
    ai.setDirectOffset(0);
    ai.setDirectAlign(0);
    ai.setSignExt(false);
    return ai;
  }

  // ABIArgInfo will record the argument as being extended based on the sign of
  // it's type.
  static ABIArgInfo getExtend(clang::QualType ty, mlir::Type t = nullptr) {
    assert(ty->isIntegralOrEnumerationType() && "Unexpected QualType");
    if (ty->hasSignedIntegerRepresentation())
      return getSignExtend(ty, t);
    return getZeroExtend(ty, t);
  }
  static ABIArgInfo getExtend(mlir::Type ty, mlir::Type t = nullptr) {
    // NOTE(cir): The original can apply this method on both integers and
    // enumerations, but in CIR, these two types are one and the same. Booleans
    // will also fall into this category, but they have their own type.
    if (mlir::isa<mlir::cir::IntType>(ty) &&
        mlir::cast<mlir::cir::IntType>(ty).isSigned())
      return getSignExtend(mlir::cast<mlir::cir::IntType>(ty), t);
    return getZeroExtend(ty, t);
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(Ignore); }

  Kind getKind() const { return theKind; }
  bool isDirect() const { return theKind == Direct; }
  bool isInAlloca() const { return theKind == InAlloca; }
  bool isExtend() const { return theKind == Extend; }
  bool isIndirect() const { return theKind == Indirect; }
  bool isIndirectAliased() const { return theKind == IndirectAliased; }
  bool isExpand() const { return theKind == Expand; }
  bool isCoerceAndExpand() const { return theKind == CoerceAndExpand; }

  bool isSignExt() const {
    assert(isExtend() && "Invalid kind!");
    return signExt;
  }
  void setSignExt(bool sExt) {
    assert(isExtend() && "Invalid kind!");
    signExt = sExt;
  }

  bool getInReg() const {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    return inReg;
  }
  void setInReg(bool ir) {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    inReg = ir;
  }

  bool canHaveCoerceToType() const {
    return isDirect() || isExtend() || isCoerceAndExpand();
  }

  // Direct/Extend accessors
  unsigned getDirectOffset() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return directAttr.offset;
  }

  void setDirectOffset(unsigned offset) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    directAttr.offset = offset;
  }

  void setDirectAlign(unsigned align) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    directAttr.align = align;
  }

  void setCanBeFlattened(bool flatten) {
    assert(isDirect() && "Invalid kind!");
    canBeFlattened = flatten;
  }

  bool getCanBeFlattened() const {
    assert(isDirect() && "Invalid kind!");
    return canBeFlattened;
  }

  mlir::Type getPaddingType() const {
    return (canHavePaddingType() ? paddingType : nullptr);
  }

  mlir::Type getCoerceToType() const {
    assert(canHaveCoerceToType() && "Invalid kind!");
    return typeData;
  }

  void setCoerceToType(mlir::Type t) {
    assert(canHaveCoerceToType() && "Invalid kind!");
    typeData = t;
  }
};

} // namespace cir

#endif // CIR_COMMON_ABIARGINFO_H
