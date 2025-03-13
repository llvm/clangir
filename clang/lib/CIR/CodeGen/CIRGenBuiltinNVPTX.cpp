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

// vprintf takes two args: A format string, and a pointer to a buffer containing
// the varargs.
//
// For example, the call
//
//   printf("format string", arg1, arg2, arg3);
//
// is converted into something resembling
//
//   struct Tmp {
//     Arg1 a1;
//     Arg2 a2;
//     Arg3 a3;
//   };
//   char* buf = alloca(sizeof(Tmp));
//   *(Tmp*)buf = {a1, a2, a3};
//   vprintf("format string", buf);
//
// `buf` is aligned to the max of {alignof(Arg1), ...}.  Furthermore, each of
// the args is itself aligned to its preferred alignment.
//
// Note that by the time this function runs, the arguments have already
// undergone the standard C vararg promotion (short -> int, float -> double
// etc). In this function we pack the arguments into the buffer described above.
mlir::Value packArgsIntoNVPTXFormatBuffer(CIRGenFunction &cgf,
                                          const CallArgList &args,
                                          mlir::Location loc) {
  const CIRDataLayout &dataLayout = cgf.CGM.getDataLayout();
  CIRGenBuilderTy &builder = cgf.getBuilder();

  if (args.size() <= 1)
    // If there are no arguments other than the format string,
    // pass a nullptr to vprintf.
    return builder.getNullPtr(cgf.VoidPtrTy, loc);

  llvm::SmallVector<mlir::Type, 8> argTypes;
  for (auto arg : llvm::drop_begin(args))
    argTypes.push_back(arg.getRValue(cgf, loc).getScalarVal().getType());

  // We can directly store the arguments into a struct, and the alignment
  // would automatically be correct. That's because vprintf does not
  // accept aggregates.
  mlir::Type allocaTy =
      cir::StructType::get(&cgf.getMLIRContext(), argTypes, /*packed=*/false,
                           /*padded=*/false, StructType::Struct);
  mlir::Value alloca =
      cgf.CreateTempAlloca(allocaTy, loc, "printf_args", nullptr);

  for (auto [i, arg] : llvm::enumerate(llvm::drop_begin(args))) {
    mlir::Value member =
        builder.createGetMember(loc, cir::PointerType::get(argTypes[i]), alloca,
                                /*name=*/"", /*index=*/i);
    auto preferredAlign = clang::CharUnits::fromQuantity(
        dataLayout.getPrefTypeAlign(argTypes[i]).value());
    builder.createAlignedStore(loc, arg.getRValue(cgf, loc).getScalarVal(),
                               member, preferredAlign);
  }

  return builder.createBitcast(alloca, cgf.VoidPtrTy);
}

mlir::Value
CIRGenFunction::emitNVPTXDevicePrintfCallExpr(const CallExpr *expr) {
  assert(CGM.getTriple().isNVPTX());
  CallArgList args;
  emitCallArgs(args,
               expr->getDirectCallee()->getType()->getAs<FunctionProtoType>(),
               expr->arguments(), expr->getDirectCallee());

  mlir::Location loc = getLoc(expr->getBeginLoc());

  // Except the format string, no non-scalar arguments are allowed for
  // device-side printf.
  bool hasNonScalar =
      llvm::any_of(llvm::drop_begin(args), [&](const CallArg &A) {
        return !A.getRValue(*this, loc).isScalar();
      });
  if (hasNonScalar) {
    CGM.ErrorUnsupported(expr, "non-scalar args to printf");
    return builder.getConstInt(loc, SInt32Ty, 0);
  }

  mlir::Value packedData = packArgsIntoNVPTXFormatBuffer(*this, args, loc);

  // int vprintf(char *format, void *packedData);
  auto vprintf = CGM.createRuntimeFunction(
      FuncType::get({cir::PointerType::get(SInt8Ty), VoidPtrTy}, SInt32Ty),
      "vprintf");
  auto formatString = args[0].getRValue(*this, loc).getScalarVal();
  return builder.createCallOp(loc, vprintf, {formatString, packedData})
      .getResult();
}
