//===------ CIRGenCUDARuntime.h - Interface to CUDA Runtimes -----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for CUDA CIR generation. Concrete
// subclasses of this implement code generation for specific OpenCL
// runtime libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
#define LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"

namespace clang::CIRGen {

class CIRGenFunction;
class CIRGenModule;
class FunctionArgList;
class RValue;
class ReturnValueSlot;

class CIRGenCUDARuntime {
protected:
  CIRGenModule &cgm;

public:
  CIRGenCUDARuntime(CIRGenModule &cgm) : cgm(cgm) {}
  virtual ~CIRGenCUDARuntime();

  virtual void emitDeviceStub(CIRGenFunction &cgf, cir::FuncOp fn,
                              FunctionArgList &args) = 0;

  virtual RValue emitCUDAKernelCallExpr(CIRGenFunction &cgf,
                                        const CUDAKernelCallExpr *expr,
                                        ReturnValueSlot retValue) = 0;
  virtual mlir::Operation *getKernelHandle(cir::FuncOp fn, GlobalDecl GD) = 0;
};

CIRGenCUDARuntime *CreateNVCUDARuntime(CIRGenModule &cgm);

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CIRGENCUDARUNTIME_H
