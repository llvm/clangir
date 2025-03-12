//===---- CIRGenBuiltinX86.cpp - Emit CIR for X86 builtins ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit NVPTX Builtin calls.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

mlir::Value CIRGenFunction::emitNVPTXBuiltinExpr(unsigned builtinId,
                                                 const CallExpr *expr) {
  auto getIntrinsic = [&](const char *name) {
    mlir::Type intTy = cir::IntType::get(&getMLIRContext(), 32, false);
    return builder
        .create<cir::LLVMIntrinsicCallOp>(getLoc(expr->getExprLoc()),
                                          builder.getStringAttr(name), intTy)
        .getResult();
  };
  switch (builtinId) {
  case NVPTX::BI__nvvm_read_ptx_sreg_tid_x:
    return getIntrinsic("nvvm.read.ptx.sreg.tid.x");
  case NVPTX::BI__nvvm_read_ptx_sreg_tid_y:
    return getIntrinsic("nvvm.read.ptx.sreg.tid.y");
  case NVPTX::BI__nvvm_read_ptx_sreg_tid_z:
    return getIntrinsic("nvvm.read.ptx.sreg.tid.z");
  case NVPTX::BI__nvvm_read_ptx_sreg_tid_w:
    return getIntrinsic("nvvm.read.ptx.sreg.tid.w");

  case NVPTX::BI__nvvm_read_ptx_sreg_ntid_x:
    return getIntrinsic("nvvm.read.ptx.sreg.ntid.x");
  case NVPTX::BI__nvvm_read_ptx_sreg_ntid_y:
    return getIntrinsic("nvvm.read.ptx.sreg.ntid.y");
  case NVPTX::BI__nvvm_read_ptx_sreg_ntid_z:
    return getIntrinsic("nvvm.read.ptx.sreg.ntid.z");
  case NVPTX::BI__nvvm_read_ptx_sreg_ntid_w:
    return getIntrinsic("nvvm.read.ptx.sreg.ntid.w");

  case NVPTX::BI__nvvm_read_ptx_sreg_ctaid_x:
    return getIntrinsic("nvvm.read.ptx.sreg.ctaid.x");
  case NVPTX::BI__nvvm_read_ptx_sreg_ctaid_y:
    return getIntrinsic("nvvm.read.ptx.sreg.ctaid.y");
  case NVPTX::BI__nvvm_read_ptx_sreg_ctaid_z:
    return getIntrinsic("nvvm.read.ptx.sreg.ctaid.z");
  case NVPTX::BI__nvvm_read_ptx_sreg_ctaid_w:
    return getIntrinsic("nvvm.read.ptx.sreg.ctaid.w");

  case NVPTX::BI__nvvm_read_ptx_sreg_nctaid_x:
    return getIntrinsic("nvvm.read.ptx.sreg.nctaid.x");
  case NVPTX::BI__nvvm_read_ptx_sreg_nctaid_y:
    return getIntrinsic("nvvm.read.ptx.sreg.nctaid.y");
  case NVPTX::BI__nvvm_read_ptx_sreg_nctaid_z:
    return getIntrinsic("nvvm.read.ptx.sreg.nctaid.z");
  case NVPTX::BI__nvvm_read_ptx_sreg_nctaid_w:
    return getIntrinsic("nvvm.read.ptx.sreg.nctaid.w");

  default:
    llvm_unreachable("NYI");
  }
}
