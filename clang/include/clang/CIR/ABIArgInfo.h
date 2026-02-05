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

#ifndef CLANG_CIR_ABIARGINFO_H
#define CLANG_CIR_ABIARGINFO_H

#include "mlir/IR/Types.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"

namespace cir {

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

    /// Valid only for integer argument types. Same as 'direct' but also emit a
    /// zero/sign extension attribute.
    Extend,

    /// Pass the argument indirectly via a hidden pointer with the specified
    /// alignment (0 indicates default alignment) and address space.
    Indirect,

    /// Similar to Indirect, but the pointer may be to an object that is
    /// otherwise referenced.  The object is known to not be modified through
    /// any other references for the duration of the call, and the callee must
    /// not itself modify the object.
    IndirectAliased,

    /// Ignore the argument (treat as void). Useful for void and empty
    /// structs.
    Ignore,

    /// CoerceAndExpand - Only valid for aggregate argument types. The structure
    /// should be expanded into consecutive arguments, but at the same time,
    /// the type should be coerced to a new type with the same layout.
    CoerceAndExpand,

    /// InAlloca - Pass the argument directly using the LLVM inalloca attribute.
    /// This is similar to indirect with byval, except it only applies to
    /// arguments stored in memory.
    InAlloca,

    // TODO: more argument kinds (Expand) will be added as the upstreaming
    // proceeds.
  };

private:
  mlir::Type typeData;    // canHaveCoerceToType()
  mlir::Type paddingType; // canHavePaddingType()
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
  };
  Kind theKind;
  bool canBeFlattened : 1;  // isDirect()
  bool inReg : 1;           // isDirect() || isExtend() || isIndirect()
  bool signExt : 1;         // isExtend()
  bool indirectByVal : 1;   // isIndirect()
  bool indirectRealign : 1; // isIndirect()
  bool sRetAfterThis : 1;   // isIndirect()

  bool canHavePaddingType() const {
    return isDirect() || isExtend() || isIndirect();
  }

  void setPaddingType(mlir::Type t) {
    assert(canHavePaddingType());
    paddingType = t;
  }

public:
  ABIArgInfo(Kind k = Direct)
      : typeData(nullptr), paddingType(nullptr), directAttr{0, 0}, theKind(k),
        canBeFlattened(false), inReg(false), signExt(false),
        indirectByVal(false), indirectRealign(false), sRetAfterThis(false) {}

  static ABIArgInfo getDirect(mlir::Type ty = nullptr, unsigned offset = 0,
                              mlir::Type padding = nullptr,
                              bool canBeFlattened = true, unsigned align = 0) {
    ABIArgInfo info(Direct);
    info.setCoerceToType(ty);
    info.setPaddingType(padding);
    info.setDirectOffset(offset);
    info.setDirectAlign(align);
    info.setCanBeFlattened(canBeFlattened);
    return info;
  }

  static ABIArgInfo getExtend(mlir::Type ty, mlir::Type coerceTy = nullptr) {
    // NOTE(cir): The original can apply this method on both integers and
    // enumerations, but in CIR, these two types are one and the same. Booleans
    // will also fall into this category, but they have their own type.
    if (mlir::isa<cir::IntType>(ty) && mlir::cast<cir::IntType>(ty).isSigned())
      return getSignExtend(ty, coerceTy);
    return getZeroExtend(ty, coerceTy);
  }

  static ABIArgInfo getSignExtend(mlir::Type ty,
                                  mlir::Type coerceTy = nullptr) {
    auto info = ABIArgInfo(Extend);
    info.setCoerceToType(coerceTy);
    info.setPaddingType(nullptr);
    info.setDirectOffset(0);
    info.setDirectAlign(0);
    info.setSignExt(true);
    return info;
  }

  static ABIArgInfo getZeroExtend(mlir::Type ty,
                                  mlir::Type coerceTy = nullptr) {
    assert(mlir::isa<cir::IntType>(ty) || mlir::isa<cir::BoolType>(ty));
    auto info = ABIArgInfo(Extend);
    info.setCoerceToType(coerceTy);
    info.setPaddingType(nullptr);
    info.setDirectOffset(0);
    info.setDirectAlign(0);
    info.setSignExt(false);
    return info;
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(Ignore); }

  static ABIArgInfo getIndirect(unsigned alignment, bool byVal = true,
                                bool realign = false,
                                mlir::Type padding = nullptr) {
    auto info = ABIArgInfo(Indirect);
    info.setIndirectAlign(alignment);
    info.setIndirectByVal(byVal);
    info.setIndirectRealign(realign);
    info.setSRetAfterThis(false);
    info.setPaddingType(padding);
    return info;
  }

  Kind getKind() const { return theKind; }
  bool isDirect() const { return theKind == Direct; }
  bool isExtend() const { return theKind == Extend; }
  bool isIndirect() const { return theKind == Indirect; }
  bool isIndirectAliased() const { return theKind == IndirectAliased; }
  bool isIgnore() const { return theKind == Ignore; }
  bool isCoerceAndExpand() const { return theKind == CoerceAndExpand; }
  bool isInAlloca() const { return theKind == InAlloca; }

  bool canHaveCoerceToType() const { return isDirect() || isExtend(); }

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

  bool getCanBeFlattened() const {
    assert(isDirect() && "Invalid kind!");
    return canBeFlattened;
  }

  void setCanBeFlattened(bool flatten) {
    assert(isDirect() && "Invalid kind!");
    canBeFlattened = flatten;
  }

  mlir::Type getPaddingType() const {
    return canHavePaddingType() ? paddingType : nullptr;
  }

  mlir::Type getCoerceToType() const {
    assert(canHaveCoerceToType() && "invalid kind!");
    return typeData;
  }

  void setCoerceToType(mlir::Type ty) {
    assert(canHaveCoerceToType() && "invalid kind!");
    typeData = ty;
  }

  // Extend accessors
  bool isSignExt() const {
    assert(isExtend() && "Invalid kind!");
    return signExt;
  }

  void setSignExt(bool sext) {
    assert(isExtend() && "Invalid kind!");
    signExt = sext;
  }

  // Indirect accessors
  unsigned getIndirectAlign() const {
    assert(isIndirect() && "Invalid kind!");
    return indirectAttr.align;
  }

  void setIndirectAlign(unsigned align) {
    assert(isIndirect() && "Invalid kind!");
    indirectAttr.align = align;
  }

  bool getIndirectByVal() const {
    assert(isIndirect() && "Invalid kind!");
    return indirectByVal;
  }

  void setIndirectByVal(bool byVal) {
    assert(isIndirect() && "Invalid kind!");
    indirectByVal = byVal;
  }

  bool getIndirectRealign() const {
    assert(isIndirect() && "Invalid kind!");
    return indirectRealign;
  }

  void setIndirectRealign(bool realign) {
    assert(isIndirect() && "Invalid kind!");
    indirectRealign = realign;
  }

  bool isSRetAfterThis() const {
    assert(isIndirect() && "Invalid kind!");
    return sRetAfterThis;
  }

  void setSRetAfterThis(bool afterThis) {
    assert(isIndirect() && "Invalid kind!");
    sRetAfterThis = afterThis;
  }

  bool getInReg() const {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    return inReg;
  }

  void setInReg(bool ir) {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    inReg = ir;
  }
};

} // namespace cir

#endif // CLANG_CIR_ABIARGINFO_H
