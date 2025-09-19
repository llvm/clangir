//===- CIRLoopOpInterface.cpp - Interface for CIR loop-like ops *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.cpp.inc"
#include "llvm/Support/ErrorHandling.h"

namespace cir {

/// Verify invariants of the LoopOpInterface.
llvm::LogicalResult detail::verifyLoopOpInterface(mlir::Operation *op) {
  auto loopOp = mlir::cast<LoopOpInterface>(op);
  if (!mlir::isa<ConditionOpInterface>(loopOp.getCond().back().getTerminator()))
    return op->emitOpError(
        "expected condition region to terminate with 'cir.condition'");
  return llvm::success();
}

} // namespace cir
