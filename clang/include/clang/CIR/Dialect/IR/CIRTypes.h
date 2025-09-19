//===- CIRTypes.h - MLIR CIR Types ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRTYPES_H
#define CLANG_CIR_DIALECT_IR_CIRTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIRTypeInterfaces.h"

namespace cir {
namespace detail {
struct RecordTypeStorage;
} // namespace detail

bool isValidFundamentalIntWidth(unsigned width);

// Returns true if the type is a CIR sized type.
bool isSized(mlir::Type ty);

//===----------------------------------------------------------------------===//
// AddressSpace helpers
//===----------------------------------------------------------------------===//

cir::AddressSpace toCIRAddressSpace(clang::LangAS langAS);

constexpr unsigned getAsUnsignedValue(cir::AddressSpace as) {
  return static_cast<unsigned>(as);
}

inline constexpr unsigned TargetAddressSpaceOffset =
    cir::getMaxEnumValForAddressSpace();

// Target address space is used for target-specific address spaces that are not
// part of the enum. Its value is represented as an offset from the maximum
// value of the enum. Make sure that it is always the last enum value.
static_assert(getAsUnsignedValue(cir::AddressSpace::Target) ==
                  cir::getMaxEnumValForAddressSpace(),
              "Target address space must be the last enum value");

constexpr bool isTargetAddressSpace(cir::AddressSpace as) {
  return getAsUnsignedValue(as) >= cir::getMaxEnumValForAddressSpace();
}

constexpr bool isLangAddressSpace(cir::AddressSpace as) {
  return !isTargetAddressSpace(as);
}

constexpr unsigned getTargetAddressSpaceValue(cir::AddressSpace as) {
  assert(isTargetAddressSpace(as) && "expected target address space");
  return getAsUnsignedValue(as) - TargetAddressSpaceOffset;
}

constexpr cir::AddressSpace computeTargetAddressSpace(unsigned v) {
  return static_cast<cir::AddressSpace>(v + TargetAddressSpaceOffset);
}

} // namespace cir

//===----------------------------------------------------------------------===//
// CIR Dialect Tablegen'd Types
//===----------------------------------------------------------------------===//

namespace cir {

#include "clang/CIR/Dialect/IR/CIRTypeConstraints.h.inc"

class AddressSpaceAttr;

} // namespace cir

#define GET_TYPEDEF_CLASSES
#include "clang/CIR/Dialect/IR/CIROpsTypes.h.inc"

#endif // CLANG_CIR_DIALECT_IR_CIRTYPES_H
