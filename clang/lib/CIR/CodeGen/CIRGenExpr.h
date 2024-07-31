//===--- CIRGenTypes.h - Type translation for CIR CodeGen -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> CIR expr lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CODEGENEXPR_H
#define LLVM_CLANG_LIB_CODEGEN_CODEGENEXPR_H
#include "CIRGenBuilder.h"

namespace cir {
mlir::Value buildArrayAccessOp(mlir::OpBuilder &builder,
                               mlir::Location arrayLocBegin,
                               mlir::Location arrayLocEnd, mlir::Value arrayPtr,
                               mlir::Type eltTy, mlir::Value idx,
                               bool shouldDecay);
}
#endif // LLVM_CLANG_LIB_CODEGEN_CODEGENEXPR_H
