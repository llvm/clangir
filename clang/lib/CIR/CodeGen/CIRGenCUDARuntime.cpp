//===----- CIRGenCUDARuntime.cpp - Interface to CUDA Runtimes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA CIR generation.  Concrete
// subclasses of this implement code generation for specific CUDA
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCUDARuntime.h"
#include "CIRGenFunction.h"
#include "clang/AST/ExprCXX.h"

using namespace clang;
using namespace CIRGen;

CIRGenCUDARuntime::~CIRGenCUDARuntime() {}

RValue CIRGenCUDARuntime::emitCUDAKernelCallExpr(CIRGenFunction &cgf,
                                                 const CUDAKernelCallExpr *expr,
                                                 ReturnValueSlot retValue) {
  auto builder = cgm.getBuilder();
  mlir::Location loc =
      cgf.currSrcLoc ? cgf.currSrcLoc.value() : builder.getUnknownLoc();

  cgf.emitIfOnBoolExpr(
      expr->getConfig(),
      [&](mlir::OpBuilder &b, mlir::Location l) {
        b.create<cir::YieldOp>(loc);
      },
      loc,
      [&](mlir::OpBuilder &b, mlir::Location l) {
        CIRGenCallee callee = cgf.emitCallee(expr->getCallee());
        cgf.emitCall(expr->getCallee()->getType(), callee, expr, retValue);
        b.create<cir::YieldOp>(loc);
      },
      loc);

  return RValue::get(nullptr);
}
