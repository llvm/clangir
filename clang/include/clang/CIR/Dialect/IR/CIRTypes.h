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

cir::ClangAddressSpace toCIRClangAddressSpace(clang::LangAS langAS);

/// Convert a LangAS to the appropriate address space attribute.
/// Returns ClangAddressSpaceAttr for clang/language-specific address spaces,
/// or TargetAddressSpaceAttr for target-specific address spaces.
mlir::Attribute toCIRClangAddressSpaceAttr(mlir::MLIRContext *ctx,
                                           clang::LangAS langAS);

constexpr unsigned getAsUnsignedValue(cir::ClangAddressSpace as) {
  return static_cast<unsigned>(as);
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
