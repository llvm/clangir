//===- CIROpInterfaces.h - CIR Op Interfaces --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_CIR_OP_H_
#define MLIR_INTERFACES_CIR_OP_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "clang/AST/Attr.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Mangle.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

namespace cir {} // namespace cir

/// Include the generated interface declarations.
#include "clang/CIR/Interfaces/CIROpInterfaces.h.inc"

namespace cir {} // namespace cir

// Casting specializations for CIRGlobalValueInterface to concrete types.
// OpInterfaces don't support casting to concrete types with reference return
// types, so we specialize CastInfo to return by value instead.
namespace llvm {
template <typename To>
struct CastInfo<
    To, cir::CIRGlobalValueInterface,
    std::enable_if_t<!std::is_same<To, cir::CIRGlobalValueInterface>::value>> {
  using CastReturnType = To;

  static inline bool isPossible(cir::CIRGlobalValueInterface interface) {
    return llvm::isa<To>(interface.getOperation());
  }

  static inline CastReturnType doCast(cir::CIRGlobalValueInterface interface) {
    return llvm::cast<To>(interface.getOperation());
  }

  static inline CastReturnType castFailed() { return CastReturnType(nullptr); }

  static inline CastReturnType
  doCastIfPossible(cir::CIRGlobalValueInterface interface) {
    if (!isPossible(interface))
      return castFailed();
    return doCast(interface);
  }
};
} // namespace llvm

#endif // MLIR_INTERFACES_CIR_OP_H_
