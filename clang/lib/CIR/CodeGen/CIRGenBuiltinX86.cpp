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
#include "llvm/ADT/TypeSwitch.h"
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
  return val.getDefiningOp<cir::ConstantOp>().getIntValue().getSExtValue();
}

// Convert the mask from an integer type to a vector of i1.
static mlir::Value getMaskVecValue(CIRGenFunction &cgf, mlir::Value mask,
                                   unsigned numElts, mlir::Location loc) {
  cir::VectorType maskTy =
      cir::VectorType::get(cgf.getBuilder().getSIntNTy(1),
                           cast<cir::IntType>(mask.getType()).getWidth());

  mlir::Value maskVec = cgf.getBuilder().createBitcast(mask, maskTy);

  // If we have less than 8 elements, then the starting mask was an i8 and
  // we need to extract down to the right number of elements.
  if (numElts < 8) {
    llvm::SmallVector<int64_t, 4> indices;
    for (unsigned i = 0; i != numElts; ++i)
      indices.push_back(i);
    maskVec = cgf.getBuilder().createVecShuffle(loc, maskVec, maskVec, indices);
  }

  return maskVec;
}

static mlir::Value emitX86MaskedStore(CIRGenFunction &cgf,
                                      ArrayRef<mlir::Value> ops,
                                      llvm::Align alignment,
                                      mlir::Location loc) {
  mlir::Value ptr = ops[0];

  mlir::Value maskVec = getMaskVecValue(
      cgf, ops[2], cast<cir::VectorType>(ops[1].getType()).getSize(), loc);

  return cgf.getBuilder().createMaskedStore(loc, ops[1], ptr, alignment,
                                            maskVec);
}

static mlir::Value emitX86MaskedLoad(CIRGenFunction &cgf,
                                     ArrayRef<mlir::Value> ops,
                                     llvm::Align alignment,
                                     mlir::Location loc) {
  mlir::Type ty = ops[1].getType();
  mlir::Value ptr = ops[0];
  mlir::Value maskVec =
      getMaskVecValue(cgf, ops[2], cast<cir::VectorType>(ty).getSize(), loc);

  return cgf.getBuilder().createMaskedLoad(loc, ty, ptr, alignment, maskVec,
                                           ops[1]);
}

static mlir::Value emitX86ExpandLoad(CIRGenFunction &cgf,
                                     ArrayRef<mlir::Value> ops,
                                     mlir::Location loc) {
  auto resultTy = cast<cir::VectorType>(ops[1].getType());
  mlir::Value ptr = ops[0];

  mlir::Value maskVec = getMaskVecValue(
      cgf, ops[2], cast<cir::VectorType>(resultTy).getSize(), loc);

  return cgf.getBuilder()
      .create<cir::LLVMIntrinsicCallOp>(
          loc, cgf.getBuilder().getStringAttr("masked.expandload"), resultTy,
          mlir::ValueRange{ptr, maskVec, ops[1]})
      .getResult();
}

static mlir::Value emitX86CompressStore(CIRGenFunction &cgf,
                                        ArrayRef<mlir::Value> ops,
                                        mlir::Location loc) {
  auto resultTy = cast<cir::VectorType>(ops[1].getType());
  mlir::Value ptr = ops[0];

  mlir::Value maskVec = getMaskVecValue(cgf, ops[2], resultTy.getSize(), loc);

  return cgf.getBuilder()
      .create<cir::LLVMIntrinsicCallOp>(
          loc, cgf.getBuilder().getStringAttr("masked.compressstore"),
          cgf.getBuilder().getVoidTy(), mlir::ValueRange{ops[1], ptr, maskVec})
      .getResult();
}

static mlir::Value emitX86SExtMask(CIRGenFunction &cgf, mlir::Value op,
                                   mlir::Type dstTy, mlir::Location loc) {
  unsigned numberOfElements = cast<cir::VectorType>(dstTy).getSize();
  mlir::Value mask = getMaskVecValue(cgf, op, numberOfElements, loc);

  return cgf.getBuilder().createCast(loc, cir::CastKind::integral, mask, dstTy);
}

static mlir::Value emitX86PSLLDQIByteShift(CIRGenFunction &cgf,
                                           const CallExpr *E,
                                           ArrayRef<mlir::Value> Ops) {
  auto &builder = cgf.getBuilder();
  unsigned shiftVal = getIntValueFromConstOp(Ops[1]) & 0xff;
  auto loc = cgf.getLoc(E->getExprLoc());
  auto resultType = cast<cir::VectorType>(Ops[0].getType());

  // If pslldq is shifting the vector more than 15 bytes, emit zero.
  // This matches the hardware behavior where shifting by 16+ bytes
  // clears the entire 128-bit lane.
  if (shiftVal >= 16)
    return builder.getZero(loc, resultType);

  // Builtin type is vXi64 so multiply by 8 to get bytes.
  unsigned numElts = resultType.getSize() * 8;
  assert(numElts % 16 == 0 && "Vector size must be multiple of 16 bytes");

  llvm::SmallVector<int64_t, 64> indices;

  // 256/512-bit pslldq operates on 128-bit lanes so we need to handle that
  for (unsigned l = 0; l < numElts; l += 16) {
    for (unsigned i = 0; i != 16; ++i) {
      unsigned idx = numElts + i - shiftVal;
      if (idx < numElts)
        idx -= numElts - 16; // end of lane, switch operand.
      indices.push_back(idx + l);
    }
  }

  // Cast to byte vector for shuffle operation
  auto byteVecTy = cir::VectorType::get(builder.getSInt8Ty(), numElts);
  mlir::Value byteCast = builder.createBitcast(Ops[0], byteVecTy);
  mlir::Value zero = builder.getZero(loc, byteVecTy);

  // Perform the shuffle (left shift by inserting zeros)
  mlir::Value shuffleResult =
      builder.createVecShuffle(loc, zero, byteCast, indices);

  // Cast back to original type
  return builder.createBitcast(shuffleResult, resultType);
}

static mlir::Value emitX86MaskedCompareResult(CIRGenFunction &cgf,
                                              mlir::Value cmp, unsigned numElts,
                                              mlir::Value maskIn,
                                              mlir::Location loc) {
  if (maskIn) {
    llvm_unreachable("NYI");
  }
  if (numElts < 8) {
    int64_t indices[8];
    for (unsigned i = 0; i != numElts; ++i)
      indices[i] = i;
    for (unsigned i = numElts; i != 8; ++i)
      indices[i] = i % numElts + numElts;

    // This should shuffle between cmp (first vector) and null (second vector)
    mlir::Value nullVec = cgf.getBuilder().getNullValue(cmp.getType(), loc);
    cmp = cgf.getBuilder().createVecShuffle(loc, cmp, nullVec, indices);
  }
  return cgf.getBuilder().createBitcast(
      cmp, cgf.getBuilder().getUIntNTy(std::max(numElts, 8U)));
}

static mlir::Value emitX86MaskedCompare(CIRGenFunction &cgf, unsigned cc,
                                        bool isSigned,
                                        ArrayRef<mlir::Value> ops,
                                        mlir::Location loc) {
  assert((ops.size() == 2 || ops.size() == 4) &&
         "Unexpected number of arguments");
  unsigned numElts = cast<cir::VectorType>(ops[0].getType()).getSize();
  mlir::Value cmp;

  if (cc == 3) {
    llvm_unreachable("NYI");
  } else if (cc == 7) {
    llvm_unreachable("NYI");
  } else {
    cir::CmpOpKind pred;
    switch (cc) {
    default:
      llvm_unreachable("Unknown condition code");
    case 0:
      pred = cir::CmpOpKind::eq;
      break;
    case 1:
      pred = cir::CmpOpKind::lt;
      break;
    case 2:
      pred = cir::CmpOpKind::le;
      break;
    case 4:
      pred = cir::CmpOpKind::ne;
      break;
    case 5:
      pred = cir::CmpOpKind::ge;
      break;
    case 6:
      pred = cir::CmpOpKind::gt;
      break;
    }

    auto resultTy = cgf.getBuilder().getType<cir::VectorType>(
        cgf.getBuilder().getUIntNTy(1), numElts);
    cmp = cgf.getBuilder().create<cir::VecCmpOp>(loc, resultTy, pred, ops[0],
                                                 ops[1]);
  }

  mlir::Value maskIn;
  if (ops.size() == 4)
    maskIn = ops[3];

  return emitX86MaskedCompareResult(cgf, cmp, numElts, maskIn, loc);
}

static mlir::Value emitX86ConvertToMask(CIRGenFunction &cgf, mlir::Value in,
                                        mlir::Location loc) {
  cir::ConstantOp zero = cgf.getBuilder().getNullValue(in.getType(), loc);
  return emitX86MaskedCompare(cgf, 1, true, {in, zero}, loc);
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
    return builder.getNullValue(convertType(E->getType()),
                                getLoc(E->getExprLoc()));
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

    uint64_t index =
        Ops[1].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();

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

    uint64_t index =
        Ops[2].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();

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
    Address tmp =
        CreateMemTemp(E->getArg(0)->getType(), getLoc(E->getExprLoc()));
    builder.createStore(getLoc(E->getExprLoc()), Ops[0], tmp);
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse.ldmxcsr"),
            builder.getVoidTy(), tmp.getPointer())
        .getResult();
  }
  case X86::BI_mm_getcsr:
  case X86::BI__builtin_ia32_stmxcsr: {
    Address tmp = CreateMemTemp(E->getType(), getLoc(E->getExprLoc()));
    builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr("x86.sse.stmxcsr"),
            builder.getVoidTy(), tmp.getPointer())
        .getResult();
    return builder.createLoad(getLoc(E->getExprLoc()), tmp);
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
  case X86::BI__builtin_ia32_storedqudi128_mask:
  case X86::BI__builtin_ia32_storedqusi128_mask:
  case X86::BI__builtin_ia32_storedquhi128_mask:
  case X86::BI__builtin_ia32_storedquqi128_mask:
  case X86::BI__builtin_ia32_storeupd128_mask:
  case X86::BI__builtin_ia32_storeups128_mask:
  case X86::BI__builtin_ia32_storedqudi256_mask:
  case X86::BI__builtin_ia32_storedqusi256_mask:
  case X86::BI__builtin_ia32_storedquhi256_mask:
  case X86::BI__builtin_ia32_storedquqi256_mask:
  case X86::BI__builtin_ia32_storeupd256_mask:
  case X86::BI__builtin_ia32_storeups256_mask:
  case X86::BI__builtin_ia32_storedqudi512_mask:
  case X86::BI__builtin_ia32_storedqusi512_mask:
  case X86::BI__builtin_ia32_storedquhi512_mask:
  case X86::BI__builtin_ia32_storedquqi512_mask:
  case X86::BI__builtin_ia32_storeupd512_mask:
  case X86::BI__builtin_ia32_storeups512_mask:
    return emitX86MaskedStore(*this, Ops, llvm::Align(1),
                              getLoc(E->getExprLoc()));
  case X86::BI__builtin_ia32_storesbf16128_mask:
  case X86::BI__builtin_ia32_storesh128_mask:
  case X86::BI__builtin_ia32_storess128_mask:
  case X86::BI__builtin_ia32_storesd128_mask:
    return emitX86MaskedStore(*this, Ops, llvm::Align(1),
                              getLoc(E->getExprLoc()));
  case X86::BI__builtin_ia32_cvtmask2b128:
  case X86::BI__builtin_ia32_cvtmask2b256:
  case X86::BI__builtin_ia32_cvtmask2b512:
  case X86::BI__builtin_ia32_cvtmask2w128:
  case X86::BI__builtin_ia32_cvtmask2w256:
  case X86::BI__builtin_ia32_cvtmask2w512:
  case X86::BI__builtin_ia32_cvtmask2d128:
  case X86::BI__builtin_ia32_cvtmask2d256:
  case X86::BI__builtin_ia32_cvtmask2d512:
  case X86::BI__builtin_ia32_cvtmask2q128:
  case X86::BI__builtin_ia32_cvtmask2q256:
  case X86::BI__builtin_ia32_cvtmask2q512:
    return emitX86SExtMask(*this, Ops[0], convertType(E->getType()),
                           getLoc(E->getExprLoc()));

  case X86::BI__builtin_ia32_cvtb2mask128:
  case X86::BI__builtin_ia32_cvtb2mask256:
  case X86::BI__builtin_ia32_cvtb2mask512:
  case X86::BI__builtin_ia32_cvtw2mask128:
  case X86::BI__builtin_ia32_cvtw2mask256:
  case X86::BI__builtin_ia32_cvtw2mask512:
  case X86::BI__builtin_ia32_cvtd2mask128:
  case X86::BI__builtin_ia32_cvtd2mask256:
  case X86::BI__builtin_ia32_cvtd2mask512:
  case X86::BI__builtin_ia32_cvtq2mask128:
  case X86::BI__builtin_ia32_cvtq2mask256:
  case X86::BI__builtin_ia32_cvtq2mask512:
    return emitX86ConvertToMask(*this, Ops[0], getLoc(E->getExprLoc()));
  case X86::BI__builtin_ia32_cvtdq2ps512_mask:
  case X86::BI__builtin_ia32_cvtqq2ps512_mask:
  case X86::BI__builtin_ia32_cvtqq2pd512_mask:
  case X86::BI__builtin_ia32_vcvtw2ph512_mask:
  case X86::BI__builtin_ia32_vcvtdq2ph512_mask:
  case X86::BI__builtin_ia32_vcvtqq2ph512_mask:
    llvm_unreachable("vcvtw2ph256_round_mask NYI");
  case X86::BI__builtin_ia32_cvtudq2ps512_mask:
  case X86::BI__builtin_ia32_cvtuqq2ps512_mask:
  case X86::BI__builtin_ia32_cvtuqq2pd512_mask:
  case X86::BI__builtin_ia32_vcvtuw2ph512_mask:
  case X86::BI__builtin_ia32_vcvtudq2ph512_mask:
  case X86::BI__builtin_ia32_vcvtuqq2ph512_mask:
    llvm_unreachable("vcvtuw2ph256_round_mask NYI");
  case X86::BI__builtin_ia32_vfmaddss3:
  case X86::BI__builtin_ia32_vfmaddsd3:
  case X86::BI__builtin_ia32_vfmaddsh3_mask:
  case X86::BI__builtin_ia32_vfmaddss3_mask:
  case X86::BI__builtin_ia32_vfmaddsd3_mask:
    llvm_unreachable("vfmaddss3 NYI");
  case X86::BI__builtin_ia32_vfmaddss:
  case X86::BI__builtin_ia32_vfmaddsd:
    llvm_unreachable("vfmaddss NYI");
  case X86::BI__builtin_ia32_vfmaddsh3_maskz:
  case X86::BI__builtin_ia32_vfmaddss3_maskz:
  case X86::BI__builtin_ia32_vfmaddsd3_maskz:
    llvm_unreachable("vfmaddsh3_maskz NYI");
  case X86::BI__builtin_ia32_vfmaddsh3_mask3:
  case X86::BI__builtin_ia32_vfmaddss3_mask3:
  case X86::BI__builtin_ia32_vfmaddsd3_mask3:
    llvm_unreachable("vfmaddsh3_mask3 NYI");
  case X86::BI__builtin_ia32_vfmsubsh3_mask3:
  case X86::BI__builtin_ia32_vfmsubss3_mask3:
  case X86::BI__builtin_ia32_vfmsubsd3_mask3:
    llvm_unreachable("vfmaddsh3_mask3 NYI");
  case X86::BI__builtin_ia32_vfmaddph:
  case X86::BI__builtin_ia32_vfmaddps:
  case X86::BI__builtin_ia32_vfmaddpd:
  case X86::BI__builtin_ia32_vfmaddph256:
  case X86::BI__builtin_ia32_vfmaddps256:
  case X86::BI__builtin_ia32_vfmaddpd256:
  case X86::BI__builtin_ia32_vfmaddph512_mask:
  case X86::BI__builtin_ia32_vfmaddph512_maskz:
  case X86::BI__builtin_ia32_vfmaddph512_mask3:
  case X86::BI__builtin_ia32_vfmaddbf16128:
  case X86::BI__builtin_ia32_vfmaddbf16256:
  case X86::BI__builtin_ia32_vfmaddbf16512:
  case X86::BI__builtin_ia32_vfmaddps512_mask:
  case X86::BI__builtin_ia32_vfmaddps512_maskz:
  case X86::BI__builtin_ia32_vfmaddps512_mask3:
  case X86::BI__builtin_ia32_vfmsubps512_mask3:
  case X86::BI__builtin_ia32_vfmaddpd512_mask:
  case X86::BI__builtin_ia32_vfmaddpd512_maskz:
  case X86::BI__builtin_ia32_vfmaddpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubph512_mask3:
    llvm_unreachable("vfmaddph256_round_mask3 NYI");
  case X86::BI__builtin_ia32_vfmaddsubph512_mask:
  case X86::BI__builtin_ia32_vfmaddsubph512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubph512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddph512_mask3:
  case X86::BI__builtin_ia32_vfmaddsubps512_mask:
  case X86::BI__builtin_ia32_vfmaddsubps512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubps512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddps512_mask3:
  case X86::BI__builtin_ia32_vfmaddsubpd512_mask:
  case X86::BI__builtin_ia32_vfmaddsubpd512_maskz:
  case X86::BI__builtin_ia32_vfmaddsubpd512_mask3:
  case X86::BI__builtin_ia32_vfmsubaddpd512_mask3:
    llvm_unreachable("vfmaddsubph256_round_mask3 NYI");
  case X86::BI__builtin_ia32_movdqa32store128_mask:
  case X86::BI__builtin_ia32_movdqa64store128_mask:
  case X86::BI__builtin_ia32_storeaps128_mask:
  case X86::BI__builtin_ia32_storeapd128_mask:
  case X86::BI__builtin_ia32_movdqa32store256_mask:
  case X86::BI__builtin_ia32_movdqa64store256_mask:
  case X86::BI__builtin_ia32_storeaps256_mask:
  case X86::BI__builtin_ia32_storeapd256_mask:
  case X86::BI__builtin_ia32_movdqa32store512_mask:
  case X86::BI__builtin_ia32_movdqa64store512_mask:
  case X86::BI__builtin_ia32_storeaps512_mask:
  case X86::BI__builtin_ia32_storeapd512_mask:
    return emitX86MaskedStore(
        *this, Ops,
        getContext().getTypeAlignInChars(E->getArg(1)->getType()).getAsAlign(),
        getLoc(E->getExprLoc()));

  case X86::BI__builtin_ia32_loadups128_mask:
  case X86::BI__builtin_ia32_loadups256_mask:
  case X86::BI__builtin_ia32_loadups512_mask:
  case X86::BI__builtin_ia32_loadupd128_mask:
  case X86::BI__builtin_ia32_loadupd256_mask:
  case X86::BI__builtin_ia32_loadupd512_mask:
  case X86::BI__builtin_ia32_loaddquqi128_mask:
  case X86::BI__builtin_ia32_loaddquqi256_mask:
  case X86::BI__builtin_ia32_loaddquqi512_mask:
  case X86::BI__builtin_ia32_loaddquhi128_mask:
  case X86::BI__builtin_ia32_loaddquhi256_mask:
  case X86::BI__builtin_ia32_loaddquhi512_mask:
  case X86::BI__builtin_ia32_loaddqusi128_mask:
  case X86::BI__builtin_ia32_loaddqusi256_mask:
  case X86::BI__builtin_ia32_loaddqusi512_mask:
  case X86::BI__builtin_ia32_loaddqudi128_mask:
  case X86::BI__builtin_ia32_loaddqudi256_mask:
  case X86::BI__builtin_ia32_loaddqudi512_mask:
    return emitX86MaskedLoad(*this, Ops, llvm::Align(1),
                             getLoc(E->getExprLoc()));

  case X86::BI__builtin_ia32_loadsbf16128_mask:
  case X86::BI__builtin_ia32_loadsh128_mask:
  case X86::BI__builtin_ia32_loadss128_mask:
  case X86::BI__builtin_ia32_loadsd128_mask:
    return emitX86MaskedLoad(*this, Ops, llvm::Align(1),
                             getLoc(E->getExprLoc()));

  case X86::BI__builtin_ia32_loadaps128_mask:
  case X86::BI__builtin_ia32_loadaps256_mask:
  case X86::BI__builtin_ia32_loadaps512_mask:
  case X86::BI__builtin_ia32_loadapd128_mask:
  case X86::BI__builtin_ia32_loadapd256_mask:
  case X86::BI__builtin_ia32_loadapd512_mask:
  case X86::BI__builtin_ia32_movdqa32load128_mask:
  case X86::BI__builtin_ia32_movdqa32load256_mask:
  case X86::BI__builtin_ia32_movdqa32load512_mask:
  case X86::BI__builtin_ia32_movdqa64load128_mask:
  case X86::BI__builtin_ia32_movdqa64load256_mask:
  case X86::BI__builtin_ia32_movdqa64load512_mask:
    return emitX86MaskedLoad(
        *this, Ops,
        getContext().getTypeAlignInChars(E->getArg(1)->getType()).getAsAlign(),
        getLoc(E->getExprLoc()));

  case X86::BI__builtin_ia32_expandloaddf128_mask:
  case X86::BI__builtin_ia32_expandloaddf256_mask:
  case X86::BI__builtin_ia32_expandloaddf512_mask:
  case X86::BI__builtin_ia32_expandloadsf128_mask:
  case X86::BI__builtin_ia32_expandloadsf256_mask:
  case X86::BI__builtin_ia32_expandloadsf512_mask:
  case X86::BI__builtin_ia32_expandloaddi128_mask:
  case X86::BI__builtin_ia32_expandloaddi256_mask:
  case X86::BI__builtin_ia32_expandloaddi512_mask:
  case X86::BI__builtin_ia32_expandloadsi128_mask:
  case X86::BI__builtin_ia32_expandloadsi256_mask:
  case X86::BI__builtin_ia32_expandloadsi512_mask:
  case X86::BI__builtin_ia32_expandloadhi128_mask:
  case X86::BI__builtin_ia32_expandloadhi256_mask:
  case X86::BI__builtin_ia32_expandloadhi512_mask:
  case X86::BI__builtin_ia32_expandloadqi128_mask:
  case X86::BI__builtin_ia32_expandloadqi256_mask:
  case X86::BI__builtin_ia32_expandloadqi512_mask:
    return emitX86ExpandLoad(*this, Ops, getLoc(E->getExprLoc()));

  case X86::BI__builtin_ia32_compressstoredf128_mask:
  case X86::BI__builtin_ia32_compressstoredf256_mask:
  case X86::BI__builtin_ia32_compressstoredf512_mask:
  case X86::BI__builtin_ia32_compressstoresf128_mask:
  case X86::BI__builtin_ia32_compressstoresf256_mask:
  case X86::BI__builtin_ia32_compressstoresf512_mask:
  case X86::BI__builtin_ia32_compressstoredi128_mask:
  case X86::BI__builtin_ia32_compressstoredi256_mask:
  case X86::BI__builtin_ia32_compressstoredi512_mask:
  case X86::BI__builtin_ia32_compressstoresi128_mask:
  case X86::BI__builtin_ia32_compressstoresi256_mask:
  case X86::BI__builtin_ia32_compressstoresi512_mask:
  case X86::BI__builtin_ia32_compressstorehi128_mask:
  case X86::BI__builtin_ia32_compressstorehi256_mask:
  case X86::BI__builtin_ia32_compressstorehi512_mask:
  case X86::BI__builtin_ia32_compressstoreqi128_mask:
  case X86::BI__builtin_ia32_compressstoreqi256_mask:
  case X86::BI__builtin_ia32_compressstoreqi512_mask:
    return emitX86CompressStore(*this, Ops, getLoc(E->getExprLoc()));

  case X86::BI__builtin_ia32_expanddf128_mask:
  case X86::BI__builtin_ia32_expanddf256_mask:
  case X86::BI__builtin_ia32_expanddf512_mask:
  case X86::BI__builtin_ia32_expandsf128_mask:
  case X86::BI__builtin_ia32_expandsf256_mask:
  case X86::BI__builtin_ia32_expandsf512_mask:
  case X86::BI__builtin_ia32_expanddi128_mask:
  case X86::BI__builtin_ia32_expanddi256_mask:
  case X86::BI__builtin_ia32_expanddi512_mask:
  case X86::BI__builtin_ia32_expandsi128_mask:
  case X86::BI__builtin_ia32_expandsi256_mask:
  case X86::BI__builtin_ia32_expandsi512_mask:
  case X86::BI__builtin_ia32_expandhi128_mask:
  case X86::BI__builtin_ia32_expandhi256_mask:
  case X86::BI__builtin_ia32_expandhi512_mask:
  case X86::BI__builtin_ia32_expandqi128_mask:
  case X86::BI__builtin_ia32_expandqi256_mask:
  case X86::BI__builtin_ia32_expandqi512_mask:
    llvm_unreachable("expand*_mask NYI");

  case X86::BI__builtin_ia32_compressdf128_mask:
  case X86::BI__builtin_ia32_compressdf256_mask:
  case X86::BI__builtin_ia32_compressdf512_mask:
  case X86::BI__builtin_ia32_compresssf128_mask:
  case X86::BI__builtin_ia32_compresssf256_mask:
  case X86::BI__builtin_ia32_compresssf512_mask:
  case X86::BI__builtin_ia32_compressdi128_mask:
  case X86::BI__builtin_ia32_compressdi256_mask:
  case X86::BI__builtin_ia32_compressdi512_mask:
  case X86::BI__builtin_ia32_compresssi128_mask:
  case X86::BI__builtin_ia32_compresssi256_mask:
  case X86::BI__builtin_ia32_compresssi512_mask:
  case X86::BI__builtin_ia32_compresshi128_mask:
  case X86::BI__builtin_ia32_compresshi256_mask:
  case X86::BI__builtin_ia32_compresshi512_mask:
  case X86::BI__builtin_ia32_compressqi128_mask:
  case X86::BI__builtin_ia32_compressqi256_mask:
  case X86::BI__builtin_ia32_compressqi512_mask:
    llvm_unreachable("compress*_mask NYI");

  case X86::BI__builtin_ia32_gather3div2df:
  case X86::BI__builtin_ia32_gather3div2di:
  case X86::BI__builtin_ia32_gather3div4df:
  case X86::BI__builtin_ia32_gather3div4di:
  case X86::BI__builtin_ia32_gather3div4sf:
  case X86::BI__builtin_ia32_gather3div4si:
  case X86::BI__builtin_ia32_gather3div8sf:
  case X86::BI__builtin_ia32_gather3div8si:
  case X86::BI__builtin_ia32_gather3siv2df:
  case X86::BI__builtin_ia32_gather3siv2di:
  case X86::BI__builtin_ia32_gather3siv4df:
  case X86::BI__builtin_ia32_gather3siv4di:
  case X86::BI__builtin_ia32_gather3siv4sf:
  case X86::BI__builtin_ia32_gather3siv4si:
  case X86::BI__builtin_ia32_gather3siv8sf:
  case X86::BI__builtin_ia32_gather3siv8si:
  case X86::BI__builtin_ia32_gathersiv8df:
  case X86::BI__builtin_ia32_gathersiv16sf:
  case X86::BI__builtin_ia32_gatherdiv8df:
  case X86::BI__builtin_ia32_gatherdiv16sf:
  case X86::BI__builtin_ia32_gathersiv8di:
  case X86::BI__builtin_ia32_gathersiv16si:
  case X86::BI__builtin_ia32_gatherdiv8di:
  case X86::BI__builtin_ia32_gatherdiv16si: {
    StringRef intrinsicName;
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_gather3div2df:
      intrinsicName = "x86.avx512.mask.gather3div2.df";
      break;
    case X86::BI__builtin_ia32_gather3div2di:
      intrinsicName = "x86.avx512.mask.gather3div2.di";
      break;
    case X86::BI__builtin_ia32_gather3div4df:
      intrinsicName = "x86.avx512.mask.gather3div4.df";
      break;
    case X86::BI__builtin_ia32_gather3div4di:
      intrinsicName = "x86.avx512.mask.gather3div4.di";
      break;
    case X86::BI__builtin_ia32_gather3div4sf:
      intrinsicName = "x86.avx512.mask.gather3div4.sf";
      break;
    case X86::BI__builtin_ia32_gather3div4si:
      intrinsicName = "x86.avx512.mask.gather3div4.si";
      break;
    case X86::BI__builtin_ia32_gather3div8sf:
      intrinsicName = "x86.avx512.mask.gather3div8.sf";
      break;
    case X86::BI__builtin_ia32_gather3div8si:
      intrinsicName = "x86.avx512.mask.gather3div8.si";
      break;
    case X86::BI__builtin_ia32_gather3siv2df:
      intrinsicName = "x86.avx512.mask.gather3siv2.df";
      break;
    case X86::BI__builtin_ia32_gather3siv2di:
      intrinsicName = "x86.avx512.mask.gather3siv2.di";
      break;
    case X86::BI__builtin_ia32_gather3siv4df:
      intrinsicName = "x86.avx512.mask.gather3siv4.df";
      break;
    case X86::BI__builtin_ia32_gather3siv4di:
      intrinsicName = "x86.avx512.mask.gather3siv4.di";
      break;
    case X86::BI__builtin_ia32_gather3siv4sf:
      intrinsicName = "x86.avx512.mask.gather3siv4.sf";
      break;
    case X86::BI__builtin_ia32_gather3siv4si:
      intrinsicName = "x86.avx512.mask.gather3siv4.si";
      break;
    case X86::BI__builtin_ia32_gather3siv8sf:
      intrinsicName = "x86.avx512.mask.gather3siv8.sf";
      break;
    case X86::BI__builtin_ia32_gather3siv8si:
      intrinsicName = "x86.avx512.mask.gather3siv8.si";
      break;
    case X86::BI__builtin_ia32_gathersiv8df:
      intrinsicName = "x86.avx512.mask.gather.dpd.512";
      break;
    case X86::BI__builtin_ia32_gathersiv16sf:
      intrinsicName = "x86.avx512.mask.gather.dps.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv8df:
      intrinsicName = "x86.avx512.mask.gather.qpd.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv16sf:
      intrinsicName = "x86.avx512.mask.gather.qps.512";
      break;
    case X86::BI__builtin_ia32_gathersiv8di:
      intrinsicName = "x86.avx512.mask.gather.dpq.512";
      break;
    case X86::BI__builtin_ia32_gathersiv16si:
      intrinsicName = "x86.avx512.mask.gather.dpi.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv8di:
      intrinsicName = "x86.avx512.mask.gather.qpq.512";
      break;
    case X86::BI__builtin_ia32_gatherdiv16si:
      intrinsicName = "x86.avx512.mask.gather.qpi.512";
      break;
    }

    unsigned minElts =
        std::min(cast<cir::VectorType>(Ops[0].getType()).getSize(),
                 cast<cir::VectorType>(Ops[2].getType()).getSize());
    Ops[3] = getMaskVecValue(*this, Ops[3], minElts, getLoc(E->getExprLoc()));
    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr(intrinsicName.str()),
            convertType(E->getType()), Ops)
        .getResult();
  }
  case X86::BI__builtin_ia32_scattersiv8df:
  case X86::BI__builtin_ia32_scattersiv16sf:
  case X86::BI__builtin_ia32_scatterdiv8df:
  case X86::BI__builtin_ia32_scatterdiv16sf:
  case X86::BI__builtin_ia32_scattersiv8di:
  case X86::BI__builtin_ia32_scattersiv16si:
  case X86::BI__builtin_ia32_scatterdiv8di:
  case X86::BI__builtin_ia32_scatterdiv16si:
  case X86::BI__builtin_ia32_scatterdiv2df:
  case X86::BI__builtin_ia32_scatterdiv2di:
  case X86::BI__builtin_ia32_scatterdiv4df:
  case X86::BI__builtin_ia32_scatterdiv4di:
  case X86::BI__builtin_ia32_scatterdiv4sf:
  case X86::BI__builtin_ia32_scatterdiv4si:
  case X86::BI__builtin_ia32_scatterdiv8sf:
  case X86::BI__builtin_ia32_scatterdiv8si:
  case X86::BI__builtin_ia32_scattersiv2df:
  case X86::BI__builtin_ia32_scattersiv2di:
  case X86::BI__builtin_ia32_scattersiv4df:
  case X86::BI__builtin_ia32_scattersiv4di:
  case X86::BI__builtin_ia32_scattersiv4sf:
  case X86::BI__builtin_ia32_scattersiv4si:
  case X86::BI__builtin_ia32_scattersiv8sf:
  case X86::BI__builtin_ia32_scattersiv8si: {
    llvm::StringRef intrinsicName;
    switch (BuiltinID) {
    default:
      llvm_unreachable("Unexpected builtin");
    case X86::BI__builtin_ia32_scattersiv8df:
      intrinsicName = "x86.avx512.mask.scatter.dpd.512";
      break;
    case X86::BI__builtin_ia32_scattersiv16sf:
      intrinsicName = "x86.avx512.mask.scatter.dps.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv8df:
      intrinsicName = "x86.avx512.mask.scatter.qpd.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv16sf:
      intrinsicName = "x86.avx512.mask.scatter.qps.512";
      break;
    case X86::BI__builtin_ia32_scattersiv8di:
      intrinsicName = "x86.avx512.mask.scatter.dpq.512";
      break;
    case X86::BI__builtin_ia32_scattersiv16si:
      intrinsicName = "x86.avx512.mask.scatter.dpi.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv8di:
      intrinsicName = "x86.avx512.mask.scatter.qpq.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv16si:
      intrinsicName = "x86.avx512.mask.scatter.qpi.512";
      break;
    case X86::BI__builtin_ia32_scatterdiv2df:
      intrinsicName = "x86.avx512.mask.scatterdiv2.df";
      break;
    case X86::BI__builtin_ia32_scatterdiv2di:
      intrinsicName = "x86.avx512.mask.scatterdiv2.di";
      break;
    case X86::BI__builtin_ia32_scatterdiv4df:
      intrinsicName = "x86.avx512.mask.scatterdiv4.df";
      break;
    case X86::BI__builtin_ia32_scatterdiv4di:
      intrinsicName = "x86.avx512.mask.scatterdiv4.di";
      break;
    case X86::BI__builtin_ia32_scatterdiv4sf:
      intrinsicName = "x86.avx512.mask.scatterdiv4.sf";
      break;
    case X86::BI__builtin_ia32_scatterdiv4si:
      intrinsicName = "x86.avx512.mask.scatterdiv4.si";
      break;
    case X86::BI__builtin_ia32_scatterdiv8sf:
      intrinsicName = "x86.avx512.mask.scatterdiv8.sf";
      break;
    case X86::BI__builtin_ia32_scatterdiv8si:
      intrinsicName = "x86.avx512.mask.scatterdiv8.si";
      break;
    case X86::BI__builtin_ia32_scattersiv2df:
      intrinsicName = "x86.avx512.mask.scattersiv2.df";
      break;
    case X86::BI__builtin_ia32_scattersiv2di:
      intrinsicName = "x86.avx512.mask.scattersiv2.di";
      break;
    case X86::BI__builtin_ia32_scattersiv4df:
      intrinsicName = "x86.avx512.mask.scattersiv4.df";
      break;
    case X86::BI__builtin_ia32_scattersiv4di:
      intrinsicName = "x86.avx512.mask.scattersiv4.di";
      break;
    case X86::BI__builtin_ia32_scattersiv4sf:
      intrinsicName = "x86.avx512.mask.scattersiv4.sf";
      break;
    case X86::BI__builtin_ia32_scattersiv4si:
      intrinsicName = "x86.avx512.mask.scattersiv4.si";
      break;
    case X86::BI__builtin_ia32_scattersiv8sf:
      intrinsicName = "x86.avx512.mask.scattersiv8.sf";
      break;
    case X86::BI__builtin_ia32_scattersiv8si:
      intrinsicName = "x86.avx512.mask.scattersiv8.si";
      break;
    }

    unsigned minElts =
        std::min(cast<cir::VectorType>(Ops[2].getType()).getSize(),
                 cast<cir::VectorType>(Ops[3].getType()).getSize());
    Ops[1] = getMaskVecValue(*this, Ops[1], minElts, getLoc(E->getExprLoc()));

    return builder
        .create<cir::LLVMIntrinsicCallOp>(
            getLoc(E->getExprLoc()), builder.getStringAttr(intrinsicName.str()),
            builder.getVoidTy(), Ops)
        .getResult();
  }

  case X86::BI__builtin_ia32_vextractf128_pd256:
  case X86::BI__builtin_ia32_vextractf128_ps256:
  case X86::BI__builtin_ia32_vextractf128_si256:
  case X86::BI__builtin_ia32_extract128i256:
  case X86::BI__builtin_ia32_extractf64x4_mask:
  case X86::BI__builtin_ia32_extractf32x4_mask:
  case X86::BI__builtin_ia32_extracti64x4_mask:
  case X86::BI__builtin_ia32_extracti32x4_mask:
  case X86::BI__builtin_ia32_extractf32x8_mask:
  case X86::BI__builtin_ia32_extracti32x8_mask:
  case X86::BI__builtin_ia32_extractf32x4_256_mask:
  case X86::BI__builtin_ia32_extracti32x4_256_mask:
  case X86::BI__builtin_ia32_extractf64x2_256_mask:
  case X86::BI__builtin_ia32_extracti64x2_256_mask:
  case X86::BI__builtin_ia32_extractf64x2_512_mask:
  case X86::BI__builtin_ia32_extracti64x2_512_mask:
    llvm_unreachable("extractf128 NYI");
  case X86::BI__builtin_ia32_vinsertf128_pd256:
  case X86::BI__builtin_ia32_vinsertf128_ps256:
  case X86::BI__builtin_ia32_vinsertf128_si256:
  case X86::BI__builtin_ia32_insert128i256:
  case X86::BI__builtin_ia32_insertf64x4:
  case X86::BI__builtin_ia32_insertf32x4:
  case X86::BI__builtin_ia32_inserti64x4:
  case X86::BI__builtin_ia32_inserti32x4:
  case X86::BI__builtin_ia32_insertf32x8:
  case X86::BI__builtin_ia32_inserti32x8:
  case X86::BI__builtin_ia32_insertf32x4_256:
  case X86::BI__builtin_ia32_inserti32x4_256:
  case X86::BI__builtin_ia32_insertf64x2_256:
  case X86::BI__builtin_ia32_inserti64x2_256:
  case X86::BI__builtin_ia32_insertf64x2_512:
  case X86::BI__builtin_ia32_inserti64x2_512: {
    unsigned dstNumElts = cast<cir::VectorType>(Ops[0].getType()).getSize();
    unsigned srcNumElts = cast<cir::VectorType>(Ops[1].getType()).getSize();
    unsigned subVectors = dstNumElts / srcNumElts;
    unsigned index =
        Ops[2].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();
    assert(llvm::isPowerOf2_32(subVectors) && "Expected power of 2 subvectors");
    index &= subVectors - 1; // Remove any extra bits.
    index *= srcNumElts;

    int64_t indices[16];
    for (unsigned i = 0; i != dstNumElts; ++i)
      indices[i] = (i >= srcNumElts) ? srcNumElts + (i % srcNumElts) : i;

    mlir::Value op1 = builder.createVecShuffle(getLoc(E->getExprLoc()), Ops[1],
                                               ArrayRef(indices, dstNumElts));

    for (unsigned i = 0; i != dstNumElts; ++i) {
      if (i >= index && i < (index + srcNumElts))
        indices[i] = (i - index) + dstNumElts;
      else
        indices[i] = i;
    }

    return builder.createVecShuffle(getLoc(E->getExprLoc()), Ops[0], op1,
                                    ArrayRef(indices, dstNumElts));
  }

  case X86::BI__builtin_ia32_pmovqd512_mask:
  case X86::BI__builtin_ia32_pmovwb512_mask:
    llvm_unreachable("pmovqd512_mask NYI");
  case X86::BI__builtin_ia32_pblendw128:
  case X86::BI__builtin_ia32_blendpd:
  case X86::BI__builtin_ia32_blendps:
  case X86::BI__builtin_ia32_blendpd256:
  case X86::BI__builtin_ia32_blendps256:
  case X86::BI__builtin_ia32_pblendw256:
  case X86::BI__builtin_ia32_pblendd128:
  case X86::BI__builtin_ia32_pblendd256: {
    unsigned numElts = cast<cir::VectorType>(Ops[0].getType()).getSize();
    unsigned imm =
        Ops[2].getDefiningOp<cir::ConstantOp>().getIntValue().getZExtValue();

    int64_t indices[16];
    // If there are more than 8 elements, the immediate is used twice so make
    // sure we handle that.
    for (unsigned i = 0; i != numElts; ++i)
      indices[i] = ((imm >> (i % 8)) & 0x1) ? numElts + i : i;

    return builder.createVecShuffle(getLoc(E->getExprLoc()), Ops[0], Ops[1],
                                    ArrayRef(indices, numElts));
  }
  case X86::BI__builtin_ia32_pshuflw:
  case X86::BI__builtin_ia32_pshuflw256:
  case X86::BI__builtin_ia32_pshuflw512:
    llvm_unreachable("pshuflw NYI");
  case X86::BI__builtin_ia32_pshufhw:
  case X86::BI__builtin_ia32_pshufhw256:
  case X86::BI__builtin_ia32_pshufhw512:
    llvm_unreachable("pshufhw NYI");
  case X86::BI__builtin_ia32_pshufd:
  case X86::BI__builtin_ia32_pshufd256:
  case X86::BI__builtin_ia32_pshufd512:
  case X86::BI__builtin_ia32_vpermilpd:
  case X86::BI__builtin_ia32_vpermilps:
  case X86::BI__builtin_ia32_vpermilpd256:
  case X86::BI__builtin_ia32_vpermilps256:
  case X86::BI__builtin_ia32_vpermilpd512:
  case X86::BI__builtin_ia32_vpermilps512: {
    uint32_t imm = getIntValueFromConstOp(Ops[1]);
    auto vecTy = cast<cir::VectorType>(Ops[0].getType());
    unsigned numElts = vecTy.getSize();
    auto eltTy = vecTy.getElementType();

    unsigned eltBitWidth = getTypeSizeInBits(eltTy).getFixedValue();
    unsigned vecBitWidth = numElts * eltBitWidth;
    unsigned numLanes = vecBitWidth / 128;
    unsigned numLaneElts = numElts / numLanes;

    imm = (imm & 0xff) * 0x01010101;
    llvm::SmallVector<int64_t, 16> indices;
    for (unsigned l = 0; l != numElts; l += numLaneElts) {
      for (unsigned i = 0; i != numLaneElts; ++i) {
        indices.push_back((imm % numLaneElts) + l);
        imm /= numLaneElts;
      }
    }
    return builder.createVecShuffle(getLoc(E->getExprLoc()), Ops[0], indices);
  }
  case X86::BI__builtin_ia32_shufpd:
  case X86::BI__builtin_ia32_shufpd256:
  case X86::BI__builtin_ia32_shufpd512:
  case X86::BI__builtin_ia32_shufps:
  case X86::BI__builtin_ia32_shufps256:
  case X86::BI__builtin_ia32_shufps512:
    llvm_unreachable("shufpd NYI");
  case X86::BI__builtin_ia32_permdi256:
  case X86::BI__builtin_ia32_permdf256:
  case X86::BI__builtin_ia32_permdi512:
  case X86::BI__builtin_ia32_permdf512:
    llvm_unreachable("permdi NYI");
  case X86::BI__builtin_ia32_palignr128:
  case X86::BI__builtin_ia32_palignr256:
  case X86::BI__builtin_ia32_palignr512:
    llvm_unreachable("palignr NYI");
  case X86::BI__builtin_ia32_alignd128:
  case X86::BI__builtin_ia32_alignd256:
  case X86::BI__builtin_ia32_alignd512:
  case X86::BI__builtin_ia32_alignq128:
  case X86::BI__builtin_ia32_alignq256:
  case X86::BI__builtin_ia32_alignq512:
    llvm_unreachable("alignd NYI");
  case X86::BI__builtin_ia32_shuf_f32x4_256:
  case X86::BI__builtin_ia32_shuf_f64x2_256:
  case X86::BI__builtin_ia32_shuf_i32x4_256:
  case X86::BI__builtin_ia32_shuf_i64x2_256:
  case X86::BI__builtin_ia32_shuf_f32x4:
  case X86::BI__builtin_ia32_shuf_f64x2:
  case X86::BI__builtin_ia32_shuf_i32x4:
  case X86::BI__builtin_ia32_shuf_i64x2:
    llvm_unreachable("shuf_f32x4 NYI");
  case X86::BI__builtin_ia32_vperm2f128_pd256:
  case X86::BI__builtin_ia32_vperm2f128_ps256:
  case X86::BI__builtin_ia32_vperm2f128_si256:
  case X86::BI__builtin_ia32_permti256:
    llvm_unreachable("vperm2f128 NYI");
  case X86::BI__builtin_ia32_pslldqi128_byteshift:
  case X86::BI__builtin_ia32_pslldqi256_byteshift:
  case X86::BI__builtin_ia32_pslldqi512_byteshift:
    return emitX86PSLLDQIByteShift(*this, E, Ops);
  case X86::BI__builtin_ia32_psrldqi128_byteshift:
  case X86::BI__builtin_ia32_psrldqi256_byteshift:
  case X86::BI__builtin_ia32_psrldqi512_byteshift:
    llvm_unreachable("psrldqi NYI");
  case X86::BI__builtin_ia32_kshiftliqi:
  case X86::BI__builtin_ia32_kshiftlihi:
  case X86::BI__builtin_ia32_kshiftlisi:
  case X86::BI__builtin_ia32_kshiftlidi:
    llvm_unreachable("kshiftl NYI");
  case X86::BI__builtin_ia32_kshiftriqi:
  case X86::BI__builtin_ia32_kshiftrihi:
  case X86::BI__builtin_ia32_kshiftrisi:
  case X86::BI__builtin_ia32_kshiftridi:
    llvm_unreachable("kshiftr NYI");
  // Rotate is a special case of funnel shift - 1st 2 args are the same.
  case X86::BI__builtin_ia32_vprotb:
  case X86::BI__builtin_ia32_vprotw:
  case X86::BI__builtin_ia32_vprotd:
  case X86::BI__builtin_ia32_vprotq:
  case X86::BI__builtin_ia32_vprotbi:
  case X86::BI__builtin_ia32_vprotwi:
  case X86::BI__builtin_ia32_vprotdi:
  case X86::BI__builtin_ia32_vprotqi:
  case X86::BI__builtin_ia32_prold128:
  case X86::BI__builtin_ia32_prold256:
  case X86::BI__builtin_ia32_prold512:
  case X86::BI__builtin_ia32_prolq128:
  case X86::BI__builtin_ia32_prolq256:
  case X86::BI__builtin_ia32_prolq512:
  case X86::BI__builtin_ia32_prolvd128:
  case X86::BI__builtin_ia32_prolvd256:
  case X86::BI__builtin_ia32_prolvd512:
  case X86::BI__builtin_ia32_prolvq128:
  case X86::BI__builtin_ia32_prolvq256:
  case X86::BI__builtin_ia32_prolvq512:
    llvm_unreachable("rotate NYI");
  case X86::BI__builtin_ia32_prord128:
  case X86::BI__builtin_ia32_prord256:
  case X86::BI__builtin_ia32_prord512:
  case X86::BI__builtin_ia32_prorq128:
  case X86::BI__builtin_ia32_prorq256:
  case X86::BI__builtin_ia32_prorq512:
  case X86::BI__builtin_ia32_prorvd128:
  case X86::BI__builtin_ia32_prorvd256:
  case X86::BI__builtin_ia32_prorvd512:
  case X86::BI__builtin_ia32_prorvq128:
  case X86::BI__builtin_ia32_prorvq256:
  case X86::BI__builtin_ia32_prorvq512:
    llvm_unreachable("prord NYI");
  case X86::BI__builtin_ia32_selectb_128:
  case X86::BI__builtin_ia32_selectb_256:
  case X86::BI__builtin_ia32_selectb_512:
  case X86::BI__builtin_ia32_selectw_128:
  case X86::BI__builtin_ia32_selectw_256:
  case X86::BI__builtin_ia32_selectw_512:
  case X86::BI__builtin_ia32_selectd_128:
  case X86::BI__builtin_ia32_selectd_256:
  case X86::BI__builtin_ia32_selectd_512:
  case X86::BI__builtin_ia32_selectq_128:
  case X86::BI__builtin_ia32_selectq_256:
  case X86::BI__builtin_ia32_selectq_512:
  case X86::BI__builtin_ia32_selectph_128:
  case X86::BI__builtin_ia32_selectph_256:
  case X86::BI__builtin_ia32_selectph_512:
  case X86::BI__builtin_ia32_selectpbf_128:
  case X86::BI__builtin_ia32_selectpbf_256:
  case X86::BI__builtin_ia32_selectpbf_512:
  case X86::BI__builtin_ia32_selectps_128:
  case X86::BI__builtin_ia32_selectps_256:
  case X86::BI__builtin_ia32_selectps_512:
  case X86::BI__builtin_ia32_selectpd_128:
  case X86::BI__builtin_ia32_selectpd_256:
  case X86::BI__builtin_ia32_selectpd_512:
    llvm_unreachable("select NYI");
  case X86::BI__builtin_ia32_selectsh_128:
  case X86::BI__builtin_ia32_selectsbf_128:
  case X86::BI__builtin_ia32_selectss_128:
  case X86::BI__builtin_ia32_selectsd_128:
    llvm_unreachable("selectsh NYI");
  case X86::BI__builtin_ia32_cmpb128_mask:
  case X86::BI__builtin_ia32_cmpb256_mask:
  case X86::BI__builtin_ia32_cmpb512_mask:
  case X86::BI__builtin_ia32_cmpw128_mask:
  case X86::BI__builtin_ia32_cmpw256_mask:
  case X86::BI__builtin_ia32_cmpw512_mask:
  case X86::BI__builtin_ia32_cmpd128_mask:
  case X86::BI__builtin_ia32_cmpd256_mask:
  case X86::BI__builtin_ia32_cmpd512_mask:
  case X86::BI__builtin_ia32_cmpq128_mask:
  case X86::BI__builtin_ia32_cmpq256_mask:
  case X86::BI__builtin_ia32_cmpq512_mask:
    llvm_unreachable("cmpb NYI");
  case X86::BI__builtin_ia32_ucmpb128_mask:
  case X86::BI__builtin_ia32_ucmpb256_mask:
  case X86::BI__builtin_ia32_ucmpb512_mask:
  case X86::BI__builtin_ia32_ucmpw128_mask:
  case X86::BI__builtin_ia32_ucmpw256_mask:
  case X86::BI__builtin_ia32_ucmpw512_mask:
  case X86::BI__builtin_ia32_ucmpd128_mask:
  case X86::BI__builtin_ia32_ucmpd256_mask:
  case X86::BI__builtin_ia32_ucmpd512_mask:
  case X86::BI__builtin_ia32_ucmpq128_mask:
  case X86::BI__builtin_ia32_ucmpq256_mask:
  case X86::BI__builtin_ia32_ucmpq512_mask:
    llvm_unreachable("ucmpb NYI");
  case X86::BI__builtin_ia32_vpcomb:
  case X86::BI__builtin_ia32_vpcomw:
  case X86::BI__builtin_ia32_vpcomd:
  case X86::BI__builtin_ia32_vpcomq:
    llvm_unreachable("vpcomb NYI");
  case X86::BI__builtin_ia32_vpcomub:
  case X86::BI__builtin_ia32_vpcomuw:
  case X86::BI__builtin_ia32_vpcomud:
  case X86::BI__builtin_ia32_vpcomuq:
    llvm_unreachable("vpcomub NYI");

  case X86::BI__builtin_ia32_kortestcqi:
  case X86::BI__builtin_ia32_kortestchi:
  case X86::BI__builtin_ia32_kortestcsi:
  case X86::BI__builtin_ia32_kortestcdi:
    llvm_unreachable("kortestc NYI");
  case X86::BI__builtin_ia32_kortestzqi:
  case X86::BI__builtin_ia32_kortestzhi:
  case X86::BI__builtin_ia32_kortestzsi:
  case X86::BI__builtin_ia32_kortestzdi:
    llvm_unreachable("kortestz NYI");
  case X86::BI__builtin_ia32_ktestcqi:
  case X86::BI__builtin_ia32_ktestzqi:
  case X86::BI__builtin_ia32_ktestchi:
  case X86::BI__builtin_ia32_ktestzhi:
  case X86::BI__builtin_ia32_ktestcsi:
  case X86::BI__builtin_ia32_ktestzsi:
  case X86::BI__builtin_ia32_ktestcdi:
  case X86::BI__builtin_ia32_ktestzdi:
    llvm_unreachable("ktestc NYI");
  case X86::BI__builtin_ia32_kaddqi:
  case X86::BI__builtin_ia32_kaddhi:
  case X86::BI__builtin_ia32_kaddsi:
  case X86::BI__builtin_ia32_kadddi:
    llvm_unreachable("kadd NYI");
  case X86::BI__builtin_ia32_kandqi:
  case X86::BI__builtin_ia32_kandhi:
  case X86::BI__builtin_ia32_kandsi:
  case X86::BI__builtin_ia32_kanddi:
    llvm_unreachable("kand NYI");
  case X86::BI__builtin_ia32_kandnqi:
  case X86::BI__builtin_ia32_kandnhi:
  case X86::BI__builtin_ia32_kandnsi:
  case X86::BI__builtin_ia32_kandndi:
    llvm_unreachable("kandn NYI");
  case X86::BI__builtin_ia32_korqi:
  case X86::BI__builtin_ia32_korhi:
  case X86::BI__builtin_ia32_korsi:
  case X86::BI__builtin_ia32_kordi:
    llvm_unreachable("kor NYI");
  case X86::BI__builtin_ia32_kxnorqi:
  case X86::BI__builtin_ia32_kxnorhi:
  case X86::BI__builtin_ia32_kxnorsi:
  case X86::BI__builtin_ia32_kxnordi:
    llvm_unreachable("kxnor NYI");
  case X86::BI__builtin_ia32_kxorqi:
  case X86::BI__builtin_ia32_kxorhi:
  case X86::BI__builtin_ia32_kxorsi:
  case X86::BI__builtin_ia32_kxordi:
    llvm_unreachable("kxor NYI");
  case X86::BI__builtin_ia32_knotqi:
  case X86::BI__builtin_ia32_knothi:
  case X86::BI__builtin_ia32_knotsi:
  case X86::BI__builtin_ia32_knotdi:
    llvm_unreachable("knot NYI");
  case X86::BI__builtin_ia32_kmovb:
  case X86::BI__builtin_ia32_kmovw:
  case X86::BI__builtin_ia32_kmovd:
  case X86::BI__builtin_ia32_kmovq:
    llvm_unreachable("kmov NYI");

  case X86::BI__builtin_ia32_kunpckdi:
  case X86::BI__builtin_ia32_kunpcksi:
  case X86::BI__builtin_ia32_kunpckhi:
    llvm_unreachable("kunpckdi NYI");

  case X86::BI__builtin_ia32_vplzcntd_128:
  case X86::BI__builtin_ia32_vplzcntd_256:
  case X86::BI__builtin_ia32_vplzcntd_512:
  case X86::BI__builtin_ia32_vplzcntq_128:
  case X86::BI__builtin_ia32_vplzcntq_256:
  case X86::BI__builtin_ia32_vplzcntq_512:
    llvm_unreachable("vplzcntd NYI");
  case X86::BI__builtin_ia32_sqrtsh_round_mask:
  case X86::BI__builtin_ia32_sqrtsd_round_mask:
  case X86::BI__builtin_ia32_sqrtss_round_mask:
    llvm_unreachable("sqrtsh_round_mask NYI");
  case X86::BI__builtin_ia32_sqrtpd256:
  case X86::BI__builtin_ia32_sqrtpd:
  case X86::BI__builtin_ia32_sqrtps256:
  case X86::BI__builtin_ia32_sqrtps:
  case X86::BI__builtin_ia32_sqrtph256:
  case X86::BI__builtin_ia32_sqrtph:
  case X86::BI__builtin_ia32_sqrtph512:
  case X86::BI__builtin_ia32_vsqrtbf16256:
  case X86::BI__builtin_ia32_vsqrtbf16:
  case X86::BI__builtin_ia32_vsqrtbf16512:
  case X86::BI__builtin_ia32_sqrtps512:
  case X86::BI__builtin_ia32_sqrtpd512:
    llvm_unreachable("sqrtps NYI");

  case X86::BI__builtin_ia32_pmuludq128:
  case X86::BI__builtin_ia32_pmuludq256:
  case X86::BI__builtin_ia32_pmuludq512:
    llvm_unreachable("pmuludq NYI");

  case X86::BI__builtin_ia32_pmuldq128:
  case X86::BI__builtin_ia32_pmuldq256:
  case X86::BI__builtin_ia32_pmuldq512:
    llvm_unreachable("pmuldq NYI");

  case X86::BI__builtin_ia32_pternlogd512_mask:
  case X86::BI__builtin_ia32_pternlogq512_mask:
  case X86::BI__builtin_ia32_pternlogd128_mask:
  case X86::BI__builtin_ia32_pternlogd256_mask:
  case X86::BI__builtin_ia32_pternlogq128_mask:
  case X86::BI__builtin_ia32_pternlogq256_mask:
    llvm_unreachable("pternlogd NYI");

  case X86::BI__builtin_ia32_pternlogd512_maskz:
  case X86::BI__builtin_ia32_pternlogq512_maskz:
  case X86::BI__builtin_ia32_pternlogd128_maskz:
  case X86::BI__builtin_ia32_pternlogd256_maskz:
  case X86::BI__builtin_ia32_pternlogq128_maskz:
  case X86::BI__builtin_ia32_pternlogq256_maskz:
    llvm_unreachable("pternlogd_maskz NYI");

  case X86::BI__builtin_ia32_vpshldd128:
  case X86::BI__builtin_ia32_vpshldd256:
  case X86::BI__builtin_ia32_vpshldd512:
  case X86::BI__builtin_ia32_vpshldq128:
  case X86::BI__builtin_ia32_vpshldq256:
  case X86::BI__builtin_ia32_vpshldq512:
  case X86::BI__builtin_ia32_vpshldw128:
  case X86::BI__builtin_ia32_vpshldw256:
  case X86::BI__builtin_ia32_vpshldw512:
    llvm_unreachable("vpshldd NYI");

  case X86::BI__builtin_ia32_vpshrdd128:
  case X86::BI__builtin_ia32_vpshrdd256:
  case X86::BI__builtin_ia32_vpshrdd512:
  case X86::BI__builtin_ia32_vpshrdq128:
  case X86::BI__builtin_ia32_vpshrdq256:
  case X86::BI__builtin_ia32_vpshrdq512:
  case X86::BI__builtin_ia32_vpshrdw128:
  case X86::BI__builtin_ia32_vpshrdw256:
  case X86::BI__builtin_ia32_vpshrdw512:
    llvm_unreachable("vpshrdd NYI");

  case X86::BI__builtin_ia32_vpshldvd128:
  case X86::BI__builtin_ia32_vpshldvd256:
  case X86::BI__builtin_ia32_vpshldvd512:
  case X86::BI__builtin_ia32_vpshldvq128:
  case X86::BI__builtin_ia32_vpshldvq256:
  case X86::BI__builtin_ia32_vpshldvq512:
  case X86::BI__builtin_ia32_vpshldvw128:
  case X86::BI__builtin_ia32_vpshldvw256:
  case X86::BI__builtin_ia32_vpshldvw512:
    llvm_unreachable("vpshldvd NYI");

  case X86::BI__builtin_ia32_vpshrdvd128:
  case X86::BI__builtin_ia32_vpshrdvd256:
  case X86::BI__builtin_ia32_vpshrdvd512:
  case X86::BI__builtin_ia32_vpshrdvq128:
  case X86::BI__builtin_ia32_vpshrdvq256:
  case X86::BI__builtin_ia32_vpshrdvq512:
  case X86::BI__builtin_ia32_vpshrdvw128:
  case X86::BI__builtin_ia32_vpshrdvw256:
  case X86::BI__builtin_ia32_vpshrdvw512:
    llvm_unreachable("vpshrdvd NYI");
  case X86::BI__builtin_ia32_reduce_fadd_pd512:
  case X86::BI__builtin_ia32_reduce_fadd_ps512:
  case X86::BI__builtin_ia32_reduce_fadd_ph512:
  case X86::BI__builtin_ia32_reduce_fadd_ph256:
  case X86::BI__builtin_ia32_reduce_fadd_ph128:
    llvm_unreachable("reduce_fadd NYI");
  case X86::BI__builtin_ia32_reduce_fmul_pd512:
  case X86::BI__builtin_ia32_reduce_fmul_ps512:
  case X86::BI__builtin_ia32_reduce_fmul_ph512:
  case X86::BI__builtin_ia32_reduce_fmul_ph256:
  case X86::BI__builtin_ia32_reduce_fmul_ph128:
    llvm_unreachable("reduce_fmul NYI");
  case X86::BI__builtin_ia32_reduce_fmax_pd512:
  case X86::BI__builtin_ia32_reduce_fmax_ps512:
  case X86::BI__builtin_ia32_reduce_fmax_ph512:
  case X86::BI__builtin_ia32_reduce_fmax_ph256:
  case X86::BI__builtin_ia32_reduce_fmax_ph128:
    llvm_unreachable("reduce_fmax NYI");
  case X86::BI__builtin_ia32_reduce_fmin_pd512:
  case X86::BI__builtin_ia32_reduce_fmin_ps512:
  case X86::BI__builtin_ia32_reduce_fmin_ph512:
  case X86::BI__builtin_ia32_reduce_fmin_ph256:
  case X86::BI__builtin_ia32_reduce_fmin_ph128:
    llvm_unreachable("reduce_fmin NYI");

  case X86::BI__builtin_ia32_rdrand16_step:
  case X86::BI__builtin_ia32_rdrand32_step:
  case X86::BI__builtin_ia32_rdrand64_step:
  case X86::BI__builtin_ia32_rdseed16_step:
  case X86::BI__builtin_ia32_rdseed32_step:
  case X86::BI__builtin_ia32_rdseed64_step:
    llvm_unreachable("rdrand_step NYI");
  case X86::BI__builtin_ia32_addcarryx_u32:
  case X86::BI__builtin_ia32_addcarryx_u64:
  case X86::BI__builtin_ia32_subborrow_u32:
  case X86::BI__builtin_ia32_subborrow_u64:
    llvm_unreachable("addcarryx_u32 NYI");

  case X86::BI__builtin_ia32_fpclassps128_mask:
  case X86::BI__builtin_ia32_fpclassps256_mask:
  case X86::BI__builtin_ia32_fpclassps512_mask:
  case X86::BI__builtin_ia32_vfpclassbf16128_mask:
  case X86::BI__builtin_ia32_vfpclassbf16256_mask:
  case X86::BI__builtin_ia32_vfpclassbf16512_mask:
  case X86::BI__builtin_ia32_fpclassph128_mask:
  case X86::BI__builtin_ia32_fpclassph256_mask:
  case X86::BI__builtin_ia32_fpclassph512_mask:
  case X86::BI__builtin_ia32_fpclasspd128_mask:
  case X86::BI__builtin_ia32_fpclasspd256_mask:
  case X86::BI__builtin_ia32_fpclasspd512_mask:
    llvm_unreachable("fpclassps NYI");

  case X86::BI__builtin_ia32_vp2intersect_q_512:
  case X86::BI__builtin_ia32_vp2intersect_q_256:
  case X86::BI__builtin_ia32_vp2intersect_q_128:
  case X86::BI__builtin_ia32_vp2intersect_d_512:
  case X86::BI__builtin_ia32_vp2intersect_d_256:
  case X86::BI__builtin_ia32_vp2intersect_d_128:
    llvm_unreachable("vp2intersect NYI");
  case X86::BI__builtin_ia32_vpmultishiftqb128:
  case X86::BI__builtin_ia32_vpmultishiftqb256:
  case X86::BI__builtin_ia32_vpmultishiftqb512:
    llvm_unreachable("vpmultishiftqb NYI");

  case X86::BI__builtin_ia32_vpshufbitqmb128_mask:
  case X86::BI__builtin_ia32_vpshufbitqmb256_mask:
  case X86::BI__builtin_ia32_vpshufbitqmb512_mask:
    llvm_unreachable("vpshufbitqmb NYI");

  // packed comparison intrinsics
  case X86::BI__builtin_ia32_cmpeqps:
  case X86::BI__builtin_ia32_cmpeqpd:
    llvm_unreachable("cmpeqps NYI");
  case X86::BI__builtin_ia32_cmpltps:
  case X86::BI__builtin_ia32_cmpltpd:
    llvm_unreachable("cmpltps NYI");
  case X86::BI__builtin_ia32_cmpleps:
  case X86::BI__builtin_ia32_cmplepd:
    llvm_unreachable("cmpleps NYI");
  case X86::BI__builtin_ia32_cmpunordps:
  case X86::BI__builtin_ia32_cmpunordpd:
    llvm_unreachable("cmpunordps NYI");
  case X86::BI__builtin_ia32_cmpneqps:
  case X86::BI__builtin_ia32_cmpneqpd:
    llvm_unreachable("cmpneqps NYI");
  case X86::BI__builtin_ia32_cmpnltps:
  case X86::BI__builtin_ia32_cmpnltpd:
    llvm_unreachable("cmpnltps NYI");
  case X86::BI__builtin_ia32_cmpnleps:
  case X86::BI__builtin_ia32_cmpnlepd:
    llvm_unreachable("cmpnleps NYI");
  case X86::BI__builtin_ia32_cmpordps:
  case X86::BI__builtin_ia32_cmpordpd:
    llvm_unreachable("cmpordps NYI");
  case X86::BI__builtin_ia32_cmpph128_mask:
  case X86::BI__builtin_ia32_cmpph256_mask:
  case X86::BI__builtin_ia32_cmpph512_mask:
  case X86::BI__builtin_ia32_cmpps128_mask:
  case X86::BI__builtin_ia32_cmpps256_mask:
  case X86::BI__builtin_ia32_cmpps512_mask:
  case X86::BI__builtin_ia32_cmppd128_mask:
  case X86::BI__builtin_ia32_cmppd256_mask:
  case X86::BI__builtin_ia32_cmppd512_mask:
  case X86::BI__builtin_ia32_vcmpbf16512_mask:
  case X86::BI__builtin_ia32_vcmpbf16256_mask:
  case X86::BI__builtin_ia32_vcmpbf16128_mask:
    llvm_unreachable("cmpph NYI");
  case X86::BI__builtin_ia32_cmpps:
  case X86::BI__builtin_ia32_cmpps256:
  case X86::BI__builtin_ia32_cmppd:
  case X86::BI__builtin_ia32_cmppd256:
    llvm_unreachable("cmpps NYI");
  // SSE scalar comparison intrinsics
  case X86::BI__builtin_ia32_cmpeqss:
    llvm_unreachable("cmpeqss NYI");
  case X86::BI__builtin_ia32_cmpltss:
    llvm_unreachable("cmpltss NYI");
  case X86::BI__builtin_ia32_cmpless:
    llvm_unreachable("cmpless NYI");
  case X86::BI__builtin_ia32_cmpunordss:
    llvm_unreachable("cmpunordss NYI");
  case X86::BI__builtin_ia32_cmpneqss:
    llvm_unreachable("cmpneqss NYI");
  case X86::BI__builtin_ia32_cmpnltss:
    llvm_unreachable("cmpnltss NYI");
  case X86::BI__builtin_ia32_cmpnless:
    llvm_unreachable("cmpnless NYI");
  case X86::BI__builtin_ia32_cmpordss:
    llvm_unreachable("cmpordss NYI");
  case X86::BI__builtin_ia32_cmpeqsd:
    llvm_unreachable("cmpeqsd NYI");
  case X86::BI__builtin_ia32_cmpltsd:
    llvm_unreachable("cmpltsd NYI");
  case X86::BI__builtin_ia32_cmplesd:
    llvm_unreachable("cmplesd NYI");
  case X86::BI__builtin_ia32_cmpunordsd:
    llvm_unreachable("cmpunordsd NYI");
  case X86::BI__builtin_ia32_cmpneqsd:
    llvm_unreachable("cmpneqsd NYI");
  case X86::BI__builtin_ia32_cmpnltsd:
    llvm_unreachable("cmpnltsd NYI");
  case X86::BI__builtin_ia32_cmpnlesd:
    llvm_unreachable("cmpnlesd NYI");
  case X86::BI__builtin_ia32_cmpordsd:
    llvm_unreachable("cmpordsd NYI");

  // f16c half2float intrinsics
  case X86::BI__builtin_ia32_vcvtph2ps:
  case X86::BI__builtin_ia32_vcvtph2ps256:
  case X86::BI__builtin_ia32_vcvtph2ps_mask:
  case X86::BI__builtin_ia32_vcvtph2ps256_mask:
  case X86::BI__builtin_ia32_vcvtph2ps512_mask:
    llvm_unreachable("vcvtph2ps NYI");
  // AVX512 bf16 intrinsics
  case X86::BI__builtin_ia32_cvtneps2bf16_128_mask:
    llvm_unreachable("cvtneps2bf16_128_mask NYI");
  case X86::BI__builtin_ia32_cvtsbf162ss_32:
    llvm_unreachable("cvtsbf162ss_32 NYI");

  case X86::BI__builtin_ia32_cvtneps2bf16_256_mask:
  case X86::BI__builtin_ia32_cvtneps2bf16_512_mask:
    llvm_unreachable("cvtneps2bf16_256_mask NYI");
  case X86::BI__cpuid:
  case X86::BI__cpuidex:
    llvm_unreachable("cpuid NYI");
  case X86::BI__emul:
  case X86::BI__emulu:
    llvm_unreachable("emul NYI");
  case X86::BI__mulh:
  case X86::BI__umulh:
  case X86::BI_mul128:
  case X86::BI_umul128:
    llvm_unreachable("mulh NYI");

  case X86::BI__faststorefence:
    llvm_unreachable("faststorefence NYI");
  case X86::BI__shiftleft128:
  case X86::BI__shiftright128:
    llvm_unreachable("shiftleft128 NYI");
  case X86::BI_ReadWriteBarrier:
  case X86::BI_ReadBarrier:
  case X86::BI_WriteBarrier:
    llvm_unreachable("readwritebarrier NYI");

  case X86::BI_AddressOfReturnAddress:
    llvm_unreachable("addressofreturnaddress NYI");
  case X86::BI__stosb:
    llvm_unreachable("stosb NYI");
  // Corresponding to intrisics which will return 2 tiles (tile0_tile1).
  case X86::BI__builtin_ia32_t2rpntlvwz0_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz0rs_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz0t1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz0rst1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1rs_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1t1_internal:
  case X86::BI__builtin_ia32_t2rpntlvwz1rst1_internal:
    llvm_unreachable("t2rpntlvwz0 NYI");
  case X86::BI__ud2:
    // llvm.trap makes a ud2a instruction on x86.
    llvm_unreachable("ud2 NYI");
  case X86::BI__int2c:
    llvm_unreachable("int2c NYI");
  case X86::BI__readfsbyte:
  case X86::BI__readfsword:
  case X86::BI__readfsdword:
  case X86::BI__readfsqword:
    llvm_unreachable("readfs NYI");
  case X86::BI__readgsbyte:
  case X86::BI__readgsword:
  case X86::BI__readgsdword:
  case X86::BI__readgsqword:
    llvm_unreachable("readgs NYI");
  case X86::BI__builtin_ia32_encodekey128_u32:
    llvm_unreachable("encodekey128_u32 NYI");
  case X86::BI__builtin_ia32_encodekey256_u32:
    llvm_unreachable("encodekey256_u32 NYI");
  case X86::BI__builtin_ia32_aesenc128kl_u8:
  case X86::BI__builtin_ia32_aesdec128kl_u8:
  case X86::BI__builtin_ia32_aesenc256kl_u8:
  case X86::BI__builtin_ia32_aesdec256kl_u8:
    llvm_unreachable("aesenc128kl_u8 NYI");
  case X86::BI__builtin_ia32_aesencwide128kl_u8:
  case X86::BI__builtin_ia32_aesdecwide128kl_u8:
  case X86::BI__builtin_ia32_aesencwide256kl_u8:
  case X86::BI__builtin_ia32_aesdecwide256kl_u8:
    llvm_unreachable("aesencwide128kl_u8 NYI");
  case X86::BI__builtin_ia32_vfcmaddcph512_mask:
    llvm_unreachable("vfcmaddcph512_mask NYI");
  case X86::BI__builtin_ia32_vfmaddcph512_mask:
    llvm_unreachable("vfmaddcph512_mask NYI");
  case X86::BI__builtin_ia32_vfcmaddcsh_round_mask:
    llvm_unreachable("vfcmaddcsh_round_mask NYI");
  case X86::BI__builtin_ia32_vfmaddcsh_round_mask:
    llvm_unreachable("vfmaddcsh_round_mask NYI");
  case X86::BI__builtin_ia32_vfcmaddcsh_round_mask3:
    llvm_unreachable("vfcmaddcsh_round_mask3 NYI");
  case X86::BI__builtin_ia32_vfmaddcsh_round_mask3:
    llvm_unreachable("vfmaddcsh_round_mask3 NYI");
  case X86::BI__builtin_ia32_prefetchi:
    llvm_unreachable("prefetchi NYI");
  }
}
