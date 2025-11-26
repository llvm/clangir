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

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
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

cir::LangAddressSpace toCIRLangAddressSpace(clang::LangAS langAS);

/// Convert a LangAS to the appropriate address space attribute interface.
/// Returns a MemorySpaceAttrInterface.
mlir::ptr::MemorySpaceAttrInterface
toCIRLangAddressSpaceAttr(mlir::MLIRContext *ctx, clang::LangAS langAS);

bool isSupportedCIRMemorySpaceAttr(
    mlir::ptr::MemorySpaceAttrInterface memorySpace);

constexpr unsigned getAsUnsignedValue(cir::LangAddressSpace as) {
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
