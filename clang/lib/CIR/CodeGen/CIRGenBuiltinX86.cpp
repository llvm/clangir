//===---- CIRGenBuiltinX86.cpp - Emit CIR for X86 builtins ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit x86/x86_64 Builtin calls as CIR or a function
// call to be later resolved.
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
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

static std::optional<CIRGenFunction::MSVCIntrin>
translateX86ToMsvcIntrin(unsigned BuiltinID) {
  using MSVCIntrin = CIRGenFunction::MSVCIntrin;
  switch (BuiltinID) {
  default:
    return std::nullopt;
  case clang::X86::BI_BitScanForward:
  case clang::X86::BI_BitScanForward64:
    return MSVCIntrin::_BitScanForward;
  case clang::X86::BI_BitScanReverse:
  case clang::X86::BI_BitScanReverse64:
    return MSVCIntrin::_BitScanReverse;
  case clang::X86::BI_InterlockedAnd64:
    return MSVCIntrin::_InterlockedAnd;
  case clang::X86::BI_InterlockedCompareExchange128:
    return MSVCIntrin::_InterlockedCompareExchange128;
  case clang::X86::BI_InterlockedExchange64:
    return MSVCIntrin::_InterlockedExchange;
  case clang::X86::BI_InterlockedExchangeAdd64:
    return MSVCIntrin::_InterlockedExchangeAdd;
  case clang::X86::BI_InterlockedExchangeSub64:
    return MSVCIntrin::_InterlockedExchangeSub;
  case clang::X86::BI_InterlockedOr64:
    return MSVCIntrin::_InterlockedOr;
  case clang::X86::BI_InterlockedXor64:
    return MSVCIntrin::_InterlockedXor;
  case clang::X86::BI_InterlockedDecrement64:
    return MSVCIntrin::_InterlockedDecrement;
  case clang::X86::BI_InterlockedIncrement64:
    return MSVCIntrin::_InterlockedIncrement;
  }
  llvm_unreachable("must return from switch");
}

/// Get integer from a mlir::Value that is an int constant or a constant op.
static int64_t getIntValueFromConstOp(mlir::Value val) {
  auto constOp = mlir::cast<cir::ConstantOp>(val.getDefiningOp());
  return (mlir::cast<cir::IntAttr>(constOp.getValue()))
      .getValue()
      .getSExtValue();
}

mlir::Value CIRGenFunction::emitX86BuiltinExpr(unsigned BuiltinID,
                                               const CallExpr *E) {
  if (BuiltinID == Builtin::BI__builtin_cpu_is)
    llvm_unreachable("__builtin_cpu_is NYI");
  if (BuiltinID == Builtin::BI__builtin_cpu_supports)
    llvm_unreachable("__builtin_cpu_supports NYI");
  if (BuiltinID == Builtin::BI__builtin_cpu_init)
    llvm_unreachable("__builtin_cpu_init NYI");

  // Handle MSVC intrinsics before argument evaluation to prevent double
  // evaluation.
  if (std::optional<MSVCIntrin> MsvcIntId = translateX86ToMsvcIntrin(BuiltinID))
    llvm_unreachable("translateX86ToMsvcIntrin NYI");

  llvm::SmallVector<mlir::Value, 4> Ops;

  // Find out if any arguments are required to be integer constant expressions.
  unsigned ICEArguments = 0;
  ASTContext::GetBuiltinTypeError Error;
  getContext().GetBuiltinType(BuiltinID, Error, &ICEArguments);
  assert(Error == ASTContext::GE_None && "Should not codegen an error");

  for (unsigned i = 0, e = E->getNumArgs(); i != e; i++) {
    Ops.push_back(emitScalarOrConstFoldImmArg(ICEArguments, i, E));
  }

  switch (BuiltinID) {
  default:
    return nullptr;
  case X86::BI_mm_prefetch: {
    mlir::Value Address = builder.createPtrBitcast(Ops[0], VoidTy);

    int64_t Hint = getIntValueFromConstOp(Ops[1]);
    mlir::Value RW = builder.create<cir::ConstantOp>(
        getLoc(E->getExprLoc()),
        cir::IntAttr::get(SInt32Ty, (Hint >> 2) & 0x1));
    mlir::Value Locality = builder.create<cir::ConstantOp>(
        getLoc(E->getExprLoc()), cir::IntAttr::get(SInt32Ty, Hint & 0x3));
    mlir::Value Data = builder.create<cir::ConstantOp>(
        getLoc(E->getExprLoc()), cir::IntAttr::get(SInt32Ty, 1));
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());

    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("prefetch"), voidTy,
            mlir::ValueRange{Address, RW, Locality, Data})
        .getResult();
  }
  case X86::BI_mm_clflush: {
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse2.clflush"),
            voidTy, Ops[0])
        .getResult();
  }
  case X86::BI_mm_lfence: {
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse2.lfence"),
            voidTy)
        .getResult();
  }
  case X86::BI_mm_pause: {
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse2.pause"),
            voidTy)
        .getResult();
  }
  case X86::BI_mm_mfence: {
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse2.mfence"),
            voidTy)
        .getResult();
  }
  case X86::BI_mm_sfence: {
    mlir::Type voidTy = cir::VoidType::get(&getMLIRContext());
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse.sfence"),
            voidTy)
        .getResult();
  }
  case X86::BI__rdtsc: {
    mlir::Type intTy = cir::IntType::get(&getMLIRContext(), 64, false);
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.rdtsc"), intTy)
        .getResult();
  }
  case X86::BI__builtin_ia32_rdtscp: {
    // For rdtscp, we need to create a proper struct type to hold {i64, i32}
    cir::RecordType resTy = builder.getAnonRecordTy(
        {builder.getUInt64Ty(), builder.getUInt32Ty()}, false, false);

    auto call = builder
                    .create<cir::LLVMIntrinsicCallOp>(
                        getLoc(E->getExprLoc()),
                        builder.getStringAttr("x86.rdtscp"), resTy)
                    .getResult();

    // Store processor ID in address param
    mlir::Value pID = builder.create<cir::ExtractMemberOp>(
        getLoc(E->getExprLoc()), builder.getUInt32Ty(), call, 1);
    builder.create<cir::StoreOp>(getLoc(E->getExprLoc()), pID, Ops[0]);

    // Return the timestamp at index 0
    return builder.create<cir::ExtractMemberOp>(getLoc(E->getExprLoc()),
                                                builder.getUInt64Ty(), call, 0);
  }
  case X86::BI__builtin_ia32_lzcnt_u16:
  case X86::BI__builtin_ia32_lzcnt_u32:
  case X86::BI__builtin_ia32_lzcnt_u64: {
    mlir::Value V = builder.create<cir::ConstantOp>(
        getLoc(E->getExprLoc()), cir::BoolAttr::get(&getMLIRContext(), false));
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("ctlz"),
            Ops[0].getType(), mlir::ValueRange{Ops[0], V})
        .getResult();
  }
  case X86::BI__builtin_ia32_tzcnt_u16:
  case X86::BI__builtin_ia32_tzcnt_u32:
  case X86::BI__builtin_ia32_tzcnt_u64: {
    mlir::Value V = builder.create<cir::ConstantOp>(
        getLoc(E->getExprLoc()), cir::BoolAttr::get(&getMLIRContext(), false));
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("cttz"),
            Ops[0].getType(), mlir::ValueRange{Ops[0], V})
        .getResult();
  }
  case X86::BI__builtin_ia32_undef128:
  case X86::BI__builtin_ia32_undef256:
  case X86::BI__builtin_ia32_undef512:
    // The x86 definition of "undef" is not the same as the LLVM definition
    // (PR32176). We leave optimizing away an unnecessary zero constant to the
    // IR optimizer and backend.
    // TODO: If we had a "freeze" IR instruction to generate a fixed undef
    // value, we should use that here instead of a zero.
    llvm_unreachable("__builtin_ia32_undefXX NYI");
  case X86::BI__builtin_ia32_vec_ext_v4hi:
  case X86::BI__builtin_ia32_vec_ext_v16qi:
  case X86::BI__builtin_ia32_vec_ext_v8hi:
  case X86::BI__builtin_ia32_vec_ext_v4si:
  case X86::BI__builtin_ia32_vec_ext_v4sf:
  case X86::BI__builtin_ia32_vec_ext_v2di:
  case X86::BI__builtin_ia32_vec_ext_v32qi:
  case X86::BI__builtin_ia32_vec_ext_v16hi:
  case X86::BI__builtin_ia32_vec_ext_v8si:
  case X86::BI__builtin_ia32_vec_ext_v4di: {
    unsigned NumElts = cast<cir::VectorType>(Ops[0].getType()).getSize();

    auto constOp = cast<cir::ConstantOp>(Ops[1].getDefiningOp());
    auto intAttr = cast<cir::IntAttr>(constOp.getValue());
    uint64_t index = intAttr.getValue().getZExtValue();

    index &= NumElts - 1;

    auto indexAttr = cir::IntAttr::get(
        cir::IntType::get(&getMLIRContext(), 64, false), index);
    auto indexVal =
        builder.create<cir::ConstantOp>(getLoc(E->getExprLoc()), indexAttr);

    // These builtins exist so we can ensure the index is an ICE and in range.
    // Otherwise we could just do this in the header file.
    return builder.create<cir::VecExtractOp>(getLoc(E->getExprLoc()), Ops[0],
                                             indexVal);
  }
  case X86::BI__builtin_ia32_vec_set_v4hi:
  case X86::BI__builtin_ia32_vec_set_v16qi:
  case X86::BI__builtin_ia32_vec_set_v8hi:
  case X86::BI__builtin_ia32_vec_set_v4si:
  case X86::BI__builtin_ia32_vec_set_v2di:
  case X86::BI__builtin_ia32_vec_set_v32qi:
  case X86::BI__builtin_ia32_vec_set_v16hi:
  case X86::BI__builtin_ia32_vec_set_v8si:
  case X86::BI__builtin_ia32_vec_set_v4di: {
    unsigned NumElts = cast<cir::VectorType>(Ops[0].getType()).getSize();

    auto constOp = cast<cir::ConstantOp>(Ops[2].getDefiningOp());
    auto intAttr = cast<cir::IntAttr>(constOp.getValue());
    uint64_t index = intAttr.getValue().getZExtValue();

    index &= NumElts - 1;

    auto indexAttr = cir::IntAttr::get(
        cir::IntType::get(&getMLIRContext(), 64, false), index);
    auto indexVal =
        builder.create<cir::ConstantOp>(getLoc(E->getExprLoc()), indexAttr);

    // These builtins exist so we can ensure the index is an ICE and in range.
    // Otherwise we could just do this in the header file.
    return builder.create<cir::VecInsertOp>(getLoc(E->getExprLoc()), Ops[0],
                                            Ops[1], indexVal);
  }
  case X86::BI_mm_setcsr:
  case X86::BI__builtin_ia32_ldmxcsr: {
    llvm_unreachable("mm_setcsr NYI");
  }
  case X86::BI_mm_getcsr:
  case X86::BI__builtin_ia32_stmxcsr: {
    llvm_unreachable("mm_getcsr NYI");
  }

  case X86::BI__builtin_ia32_xsave:
  case X86::BI__builtin_ia32_xsave64:
  case X86::BI__builtin_ia32_xrstor:
  case X86::BI__builtin_ia32_xrstor64:
  case X86::BI__builtin_ia32_xsaveopt:
  case X86::BI__builtin_ia32_xsaveopt64:
  case X86::BI__builtin_ia32_xrstors:
  case X86::BI__builtin_ia32_xrstors64:
  case X86::BI__builtin_ia32_xsavec:
  case X86::BI__builtin_ia32_xsavec64:
  case X86::BI__builtin_ia32_xsaves:
  case X86::BI__builtin_ia32_xsaves64:
  case X86::BI__builtin_ia32_xsetbv:
  case X86::BI_xsetbv: {
    std::string intrinsicName;

    // TODO(cir): Refactor this once we have the proper
    // infrastructure that handles `getIntrinsic` similar to OG CodeGen.
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unsupported intrinsic!");
    case X86::BI__builtin_ia32_xsave:
      intrinsicName = "x86.xsave";
      break;
    case X86::BI__builtin_ia32_xsave64:
      intrinsicName = "x86.xsave64";
      break;
    case X86::BI__builtin_ia32_xrstor:
      intrinsicName = "x86.xrstor";
      break;
    case X86::BI__builtin_ia32_xrstor64:
      intrinsicName = "x86.xrstor64";
      break;
    case X86::BI__builtin_ia32_xsaveopt:
      intrinsicName = "x86.xsaveopt";
      break;
    case X86::BI__builtin_ia32_xsaveopt64:
      intrinsicName = "x86.xsaveopt64";
      break;
    case X86::BI__builtin_ia32_xrstors:
      intrinsicName = "x86.xrstors";
      break;
    case X86::BI__builtin_ia32_xrstors64:
      intrinsicName = "x86.xrstors64";
      break;
    case X86::BI__builtin_ia32_xsavec:
      intrinsicName = "x86.xsavec";
      break;
    case X86::BI__builtin_ia32_xsavec64:
      intrinsicName = "x86.xsavec64";
      break;
    case X86::BI__builtin_ia32_xsaves:
      intrinsicName = "x86.xsaves";
      break;
    case X86::BI__builtin_ia32_xsaves64:
      intrinsicName = "x86.xsaves64";
      break;
    case X86::BI__builtin_ia32_xsetbv:
    case X86::BI_xsetbv:
      intrinsicName = "x86.xsetbv";
      break;
    }
    auto loc = getLoc(E->getExprLoc());

    mlir::Value mhi = builder.createShift(Ops[1], 32, false);
    mhi = builder.createIntCast(mhi, builder.getSInt32Ty());

    mlir::Value mlo = builder.createIntCast(Ops[1], builder.getSInt32Ty());

    Ops[1] = mhi;
    Ops.push_back(mlo);

    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            loc, builder.getStringAttr(intrinsicName), builder.getVoidTy(), Ops)
        .getResult();
  }
  case X86::BI__builtin_ia32_xgetbv:
  case X86::BI_xgetbv:
    return builder
        .create<cir::LLVMIntrinsicCallOp>(getLoc(E->getExprLoc()),
                                          builder.getStringAttr("x86.xgetbv"),
                                          builder.getUInt64Ty(), Ops)
        .getResult();
  }
}
