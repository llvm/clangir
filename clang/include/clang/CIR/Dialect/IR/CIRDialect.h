//===- CIRDialect.h - CIR dialect -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIROpsDialect.h.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIROpsStructs.h.inc"
#include "clang/CIR/Dialect/IR/CIRTraits.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "clang/CIR/Interfaces/CIRLoopOpInterface.h"
#include "clang/CIR/Interfaces/CIROpInterfaces.h"

namespace cir {
void buildTerminatedBody(mlir::OpBuilder &builder, mlir::Location loc);
} // namespace cir

#define GET_OP_CLASSES
#include "clang/CIR/Dialect/IR/CIROps.h.inc"

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRDIALECT_H
