//===---- CIRGenBuiltinAMDGPU.cpp - Emit CIR for AMDGPU builtins ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit AMDGPU Builtin calls.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"

#include "mlir/IR/Value.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/Support/ErrorHandling.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

static llvm::StringRef getIntrinsicNameforWaveReduction(unsigned BuiltinID) {
  switch (BuiltinID) {
  default:
    llvm_unreachable("Unknown BuiltinID for wave reduction");
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u64:
    return "amdgcn.wave.reduce.add";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u64:
    return "amdgcn.wave.reduce.sub";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i64:
    return "amdgcn.wave.reduce.min";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u64:
    return "amdgcn.wave.reduce.umin";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i64:
    return "amdgcn.wave.reduce.max";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u64:
    return "amdgcn.wave.reduce.umax";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b64:
    return "amdgcn.wave.reduce.and";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b64:
    return "amdgcn.wave.reduce.or";
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b32:
  case clang::AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b64:
    return "amdgcn.wave.reduce.xor";
  }
}

static mlir::Value
emitAMDGCNImageOverloadedReturnType(CIRGenFunction &CGF, const CallExpr *E,
                                    llvm::StringRef IntrinsicName,
                                    bool IsImageStore) {
  auto &Builder = CGF.getBuilder();

  auto findTextureDescIndex = [&CGF](const CallExpr *E) -> unsigned {
    QualType TexQT = CGF.getContext().AMDGPUTextureTy;
    for (unsigned I = 0, N = E->getNumArgs(); I < N; ++I) {
      QualType ArgTy = E->getArg(I)->getType();
      if (ArgTy == TexQT ||
          ArgTy.getCanonicalType() == TexQT.getCanonicalType()) {
        return I;
      }
    }
    return ~0U;
  };

  unsigned RsrcIndex = findTextureDescIndex(E);
  if (RsrcIndex == ~0U) {
    llvm::report_fatal_error("Invalid argument count for image builtin");
  }

  cir::VectorType Vec8I32Ty = cir::VectorType::get(Builder.getSInt32Ty(), 8);

  llvm::SmallVector<mlir::Value, 10> Args;
  for (unsigned I = 0, N = E->getNumArgs(); I < N; ++I) {
    mlir::Value V = CGF.emitScalarExpr(E->getArg(I));

    if (I == RsrcIndex) {
      mlir::Type VTy = V.getType();
      if (mlir::isa<cir::PointerType>(VTy)) {
        V = Builder.createAlignedLoad(CGF.getLoc(E->getExprLoc()), Vec8I32Ty, V,
                                      CharUnits::fromQuantity(32));
      }
    }
    Args.push_back(V);
  }

  mlir::Type RetTy;
  if (IsImageStore) {
    RetTy = cir::VoidType::get(Builder.getContext());
  } else {
    RetTy = CGF.convertType(E->getType());
  }

  auto CallOp = LLVMIntrinsicCallOp::create(
      Builder, CGF.getLoc(E->getExprLoc()),
      Builder.getStringAttr(IntrinsicName), RetTy, Args);

  return CallOp.getResult();
}

// Emit an intrinsic that has 1 float or double operand, and 1 integer.
static mlir::Value emitFPIntBuiltin(CIRGenFunction &CGF, const CallExpr *E,
                                    llvm::StringRef intrinsicName) {
  mlir::Value Src0 = CGF.emitScalarExpr(E->getArg(0));
  mlir::Value Src1 = CGF.emitScalarExpr(E->getArg(1));
  mlir::Value result =
      LLVMIntrinsicCallOp::create(CGF.getBuilder(), CGF.getLoc(E->getExprLoc()),
                                  CGF.getBuilder().getStringAttr(intrinsicName),
                                  Src0.getType(), {Src0, Src1})
          .getResult();
  return result;
}

static mlir::Value emitBinaryExpMaybeConstrainedFPBuiltin(
    CIRGenFunction &CGF, const CallExpr *E, llvm::StringRef IntrinsicName,
    llvm::StringRef ConstrainedIntrinsicName) {
  mlir::Value Src0 = CGF.emitScalarExpr(E->getArg(0));
  mlir::Value Src1 = CGF.emitScalarExpr(E->getArg(1));

  auto &Builder = CGF.getBuilder();

  CIRGenFunction::CIRGenFPOptionsRAII FPOptsRAII(CGF, E);

  if (Builder.getIsFPConstrained()) {
    return cir::LLVMIntrinsicCallOp::create(
               Builder, CGF.getLoc(E->getExprLoc()),
               Builder.getStringAttr(ConstrainedIntrinsicName), Src0.getType(),
               {Src0, Src1})
        .getResult();
  }

  return cir::LLVMIntrinsicCallOp::create(Builder, CGF.getLoc(E->getExprLoc()),
                                          Builder.getStringAttr(IntrinsicName),
                                          Src0.getType(), {Src0, Src1})
      .getResult();
}

static mlir::Value emitLogbBuiltin(CIRGenFunction &CGF, const CallExpr *E,
                                   bool IsFloat) {
  auto &Builder = CGF.getBuilder();
  mlir::Location Loc = CGF.getLoc(E->getExprLoc());

  mlir::Value Src0 = CGF.emitScalarExpr(E->getArg(0));
  mlir::Type SrcTy = Src0.getType();
  mlir::Type Int32Ty = Builder.getSInt32Ty();

  cir::RecordType FrExpResTy =
      Builder.getAnonRecordTy({SrcTy, Int32Ty}, false, false);

  mlir::Value FrExpResult =
      cir::LLVMIntrinsicCallOp::create(
          Builder, Loc, Builder.getStringAttr("llvm.frexp"), FrExpResTy, {Src0})
          .getResult();

  mlir::Value Exp =
      cir::ExtractMemberOp::create(Builder, Loc, Int32Ty, FrExpResult, 1);

  mlir::Value NegativeOne =
      Builder.getConstant(Loc, cir::IntAttr::get(Int32Ty, -1));
  mlir::Value ExpMinus1 = Builder.createAdd(Exp, NegativeOne);

  mlir::Value SIToFP = cir::CastOp::create(
      Builder, Loc, SrcTy, cir::CastKind::int_to_float, ExpMinus1);

  mlir::Value Fabs = cir::FAbsOp::create(Builder, Loc, SrcTy, Src0);

  llvm::APFloat InfVal =
      IsFloat ? llvm::APFloat::getInf(llvm::APFloat::IEEEsingle())
              : llvm::APFloat::getInf(llvm::APFloat::IEEEdouble());
  mlir::Value Inf = Builder.getConstant(Loc, cir::FPAttr::get(SrcTy, InfVal));

  mlir::Value FabsNegInf =
      Builder.createCompare(Loc, cir::CmpOpKind::ne, Fabs, Inf);

  mlir::Value Sel = Builder.createSelect(Loc, FabsNegInf, SIToFP, Fabs);

  llvm::APFloat ZeroValue =
      IsFloat ? llvm::APFloat::getZero(llvm::APFloat::IEEEsingle())
              : llvm::APFloat::getZero(llvm::APFloat::IEEEdouble());
  mlir::Value Zero =
      Builder.getConstant(Loc, cir::FPAttr::get(SrcTy, ZeroValue));

  mlir::Value SrcEqZero =
      Builder.createCompare(Loc, cir::CmpOpKind::eq, Src0, Zero);

  llvm::APFloat NegInfVal =
      IsFloat ? llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), true)
              : llvm::APFloat::getInf(llvm::APFloat::IEEEdouble(), true);
  mlir::Value NegInf =
      Builder.getConstant(Loc, cir::FPAttr::get(SrcTy, NegInfVal));

  mlir::Value Result = Builder.createSelect(Loc, SrcEqZero, NegInf, Sel);

  return Result;
}

mlir::Value CIRGenFunction::emitAMDGPUBuiltinExpr(unsigned builtinId,
                                                  const CallExpr *expr) {
  switch (builtinId) {
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b32:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_add_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_sub_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_i64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_min_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_i64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_max_u64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_and_b64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_or_b64:
  case AMDGPU::BI__builtin_amdgcn_wave_reduce_xor_b64: {
    llvm::StringRef intrinsicName = getIntrinsicNameforWaveReduction(builtinId);
    mlir::Value Value = emitScalarExpr(expr->getArg(0));
    mlir::Value Strategy = emitScalarExpr(expr->getArg(1));
    return LLVMIntrinsicCallOp::create(builder, getLoc(expr->getExprLoc()),
                                       builder.getStringAttr(intrinsicName),
                                       Value.getType(), {Value, Strategy})
        .getResult();
  }
  case AMDGPU::BI__builtin_amdgcn_div_scale:
  case AMDGPU::BI__builtin_amdgcn_div_scalef: {
    Address flagOutPtr = emitPointerWithAlignment(expr->getArg(3));
    llvm::StringRef intrinsicName = "amdgcn.div.scale";
    mlir::Value x = emitScalarExpr(expr->getArg(0));
    mlir::Value y = emitScalarExpr(expr->getArg(1));
    mlir::Value z = emitScalarExpr(expr->getArg(2));

    auto i1Ty = builder.getUIntNTy(1);
    cir::RecordType resTy = builder.getAnonRecordTy(
        {x.getType(), i1Ty}, /*packed=*/false, /*padded=*/false);

    mlir::Value structResult =
        cir::LLVMIntrinsicCallOp::create(builder, getLoc(expr->getExprLoc()),
                                         builder.getStringAttr(intrinsicName),
                                         resTy, {x, y, z})
            .getResult();

    mlir::Value result = cir::ExtractMemberOp::create(
        builder, getLoc(expr->getExprLoc()), x.getType(), structResult, 0);
    mlir::Value flag = cir::ExtractMemberOp::create(
        builder, getLoc(expr->getExprLoc()), i1Ty, structResult, 1);

    mlir::Type flagType = flagOutPtr.getElementType();
    mlir::Value flagToStore =
        cir::CastOp::create(builder, getLoc(expr->getExprLoc()), flagType,
                            cir::CastKind::int_to_bool, flag);
    cir::StoreOp::create(builder, getLoc(expr->getExprLoc()), flagToStore,
                         flagOutPtr.getPointer());
    return result;
  }
  case AMDGPU::BI__builtin_amdgcn_div_fmas:
  case AMDGPU::BI__builtin_amdgcn_div_fmasf: {
    mlir::Value src0 = emitScalarExpr(expr->getArg(0));
    mlir::Value src1 = emitScalarExpr(expr->getArg(1));
    mlir::Value src2 = emitScalarExpr(expr->getArg(2));
    mlir::Value src3 = emitScalarExpr(expr->getArg(3));
    mlir::Value result =
        LLVMIntrinsicCallOp::create(builder, getLoc(expr->getExprLoc()),
                                    builder.getStringAttr("amdgcn.div.fmas"),
                                    src0.getType(), {src0, src1, src2, src3})
            .getResult();
    return result;
  }
  case AMDGPU::BI__builtin_amdgcn_ds_swizzle:
    return emitBuiltinWithOneOverloadedType<2>(expr, "amdgcn.ds.swizzle")
        .getScalarVal();
  case AMDGPU::BI__builtin_amdgcn_mov_dpp8:
  case AMDGPU::BI__builtin_amdgcn_mov_dpp:
  case AMDGPU::BI__builtin_amdgcn_update_dpp: {
    llvm_unreachable("mov_dpp_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_permlane16:
  case AMDGPU::BI__builtin_amdgcn_permlanex16: {
    llvm::StringRef intrinsicName =
        builtinId == AMDGPU::BI__builtin_amdgcn_permlane16
            ? "amdgcn.permlane16"
            : "amdgcn.permlanex16";
    return emitBuiltinWithOneOverloadedType<6>(expr, intrinsicName)
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_permlane64: {
    return emitBuiltinWithOneOverloadedType<1>(expr, "amdgcn.permlane64")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_readlane: {
    return emitBuiltinWithOneOverloadedType<2>(expr, "amdgcn.readlane")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_readfirstlane: {
    return emitBuiltinWithOneOverloadedType<1>(expr, "amdgcn.readfirstlane")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_div_fixup:
  case AMDGPU::BI__builtin_amdgcn_div_fixupf:
  case AMDGPU::BI__builtin_amdgcn_div_fixuph: {
    return emitBuiltinWithOneOverloadedType<3>(expr, "amdgcn.div.fixup")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_trig_preop:
  case AMDGPU::BI__builtin_amdgcn_trig_preopf: {
    return emitFPIntBuiltin(*this, expr, "amdgcn.trig.preop");
  }
  case AMDGPU::BI__builtin_amdgcn_rcp:
  case AMDGPU::BI__builtin_amdgcn_rcpf:
  case AMDGPU::BI__builtin_amdgcn_rcph:
  case AMDGPU::BI__builtin_amdgcn_rcp_bf16: {
    return emitBuiltinWithOneOverloadedType<1>(expr, "amdgcn.rcp")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_sqrt:
  case AMDGPU::BI__builtin_amdgcn_sqrtf:
  case AMDGPU::BI__builtin_amdgcn_sqrth:
  case AMDGPU::BI__builtin_amdgcn_sqrt_bf16: {
    return emitBuiltinWithOneOverloadedType<1>(expr, "amdgcn.sqrt")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_rsq:
  case AMDGPU::BI__builtin_amdgcn_rsqf:
  case AMDGPU::BI__builtin_amdgcn_rsqh:
  case AMDGPU::BI__builtin_amdgcn_rsq_bf16: {
    return emitBuiltinWithOneOverloadedType<1>(expr, "amdgcn.rsq")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_rsq_clamp:
  case AMDGPU::BI__builtin_amdgcn_rsq_clampf: {
    return emitBuiltinWithOneOverloadedType<1>(expr, "amdgcn.rsq.clamp")
        .getScalarVal();
  }
  case AMDGPU::BI__builtin_amdgcn_sinf:
  case AMDGPU::BI__builtin_amdgcn_sinh:
  case AMDGPU::BI__builtin_amdgcn_sin_bf16: {
    llvm_unreachable("sinf_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_cosf:
  case AMDGPU::BI__builtin_amdgcn_cosh:
  case AMDGPU::BI__builtin_amdgcn_cos_bf16: {
    llvm_unreachable("cosf_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_dispatch_ptr: {
    llvm_unreachable("dispatch_ptr_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_logf:
  case AMDGPU::BI__builtin_amdgcn_log_bf16: {
    llvm_unreachable("logf_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_exp2f:
  case AMDGPU::BI__builtin_amdgcn_exp2_bf16: {
    llvm_unreachable("exp2f_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_log_clampf: {
    llvm_unreachable("log_clampf_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_ldexp:
  case AMDGPU::BI__builtin_amdgcn_ldexpf:
  case AMDGPU::BI__builtin_amdgcn_ldexph: {
    llvm_unreachable("ldexp_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_frexp_mant:
  case AMDGPU::BI__builtin_amdgcn_frexp_mantf:
  case AMDGPU::BI__builtin_amdgcn_frexp_manth: {
    llvm_unreachable("frexp_mant_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_frexp_exp:
  case AMDGPU::BI__builtin_amdgcn_frexp_expf:
  case AMDGPU::BI__builtin_amdgcn_frexp_exph: {
    llvm_unreachable("frexp_exp_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_fract:
  case AMDGPU::BI__builtin_amdgcn_fractf:
  case AMDGPU::BI__builtin_amdgcn_fracth: {
    llvm_unreachable("fract_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_lerp: {
    llvm_unreachable("lerp_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_ubfe: {
    llvm_unreachable("ubfe_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_sbfe: {
    llvm_unreachable("sbfe_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_ballot_w32:
  case AMDGPU::BI__builtin_amdgcn_ballot_w64: {
    llvm_unreachable("ballot_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_inverse_ballot_w32:
  case AMDGPU::BI__builtin_amdgcn_inverse_ballot_w64: {
    llvm_unreachable("inverse_ballot_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_tanhf:
  case AMDGPU::BI__builtin_amdgcn_tanhh:
  case AMDGPU::BI__builtin_amdgcn_tanh_bf16: {
    llvm_unreachable("tanh_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_uicmp:
  case AMDGPU::BI__builtin_amdgcn_uicmpl:
  case AMDGPU::BI__builtin_amdgcn_sicmp:
  case AMDGPU::BI__builtin_amdgcn_sicmpl: {
    llvm_unreachable("uicmp_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_fcmp:
  case AMDGPU::BI__builtin_amdgcn_fcmpf: {
    llvm_unreachable("fcmp_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_class:
  case AMDGPU::BI__builtin_amdgcn_classf:
  case AMDGPU::BI__builtin_amdgcn_classh: {
    llvm_unreachable("class_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_fmed3f:
  case AMDGPU::BI__builtin_amdgcn_fmed3h: {
    llvm_unreachable("fmed3_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_ds_append:
  case AMDGPU::BI__builtin_amdgcn_ds_consume: {
    llvm_unreachable("ds_append_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v4bf16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr_b128_v8bf16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_global_load_tr16_b128_v8bf16: {
    llvm_unreachable("global_load_tr_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8i16:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8f16:
  case AMDGPU::BI__builtin_amdgcn_ds_load_tr16_b128_v8bf16: {
    llvm_unreachable("ds_load_tr_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr4_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr8_b64_v2i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr6_b96_v3i32:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4f16:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_read_tr16_b64_v4i16: {
    llvm_unreachable("ds_read_tr_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b32:
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b64:
  case AMDGPU::BI__builtin_amdgcn_global_load_monitor_b128:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b32:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b64:
  case AMDGPU::BI__builtin_amdgcn_flat_load_monitor_b128: {
    llvm_unreachable("global_load_monitor_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b32:
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b64:
  case AMDGPU::BI__builtin_amdgcn_cluster_load_b128: {
    llvm_unreachable("cluster_load_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_load_to_lds: {
    llvm_unreachable("load_to_lds_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_32x4B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_32x4B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_16x8B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_16x8B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_load_8x16B:
  case AMDGPU::BI__builtin_amdgcn_cooperative_atomic_store_8x16B: {
    llvm_unreachable("cooperative_atomic_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_get_fpenv:
  case AMDGPU::BI__builtin_amdgcn_set_fpenv: {
    llvm_unreachable("fpenv_* builtins NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_read_exec:
  case AMDGPU::BI__builtin_amdgcn_read_exec_lo:
  case AMDGPU::BI__builtin_amdgcn_read_exec_hi: {
    llvm_unreachable("read_exec_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_h:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_l:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_intersect_ray_lh: {
    llvm_unreachable("image_bvh_intersect_ray_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_image_bvh8_intersect_ray:
  case AMDGPU::BI__builtin_amdgcn_image_bvh_dual_intersect_ray: {
    llvm_unreachable("image_bvh8_intersect_ray_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push4_pop1_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop1_rtn:
  case AMDGPU::BI__builtin_amdgcn_ds_bvh_stack_push8_pop2_rtn: {
    llvm_unreachable("ds_bvh_stack_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_image_load_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.load.1d", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.1darray", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.load.2d", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.2darray", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.load.3d", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.load.cube", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.mip.1d", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.mip.1darray", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.mip.2d", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.mip.2darray", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.mip.3d", false);
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_load_mip_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.load.mip.cube", false);
  case AMDGPU::BI__builtin_amdgcn_image_store_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.store.1d", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.1darray", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.store.2d", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.2darray", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.store.3d", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(*this, expr,
                                               "amdgcn.image.store.cube", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.mip.1d", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_1darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.mip.1darray", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.mip.2d", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_2darray_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.mip.2darray", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_3d_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_3d_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.mip.3d", true);
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_cube_v4f32_i32:
  case AMDGPU::BI__builtin_amdgcn_image_store_mip_cube_v4f16_i32:
    return emitAMDGCNImageOverloadedReturnType(
        *this, expr, "amdgcn.image.store.mip.cube", true);
  case AMDGPU::BI__builtin_amdgcn_image_sample_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_cube_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_cube_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2d_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_3d_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_3d_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_cube_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_cube_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_cube_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_cube_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_1darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_lz_2darray_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_l_2darray_f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2darray_v4f32_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2darray_v4f16_f32:
  case AMDGPU::BI__builtin_amdgcn_image_sample_d_2darray_f32_f32: {
    llvm_unreachable("image_sample_d_2darray_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_image_gather4_lz_2d_v4f32_f32: {
    llvm_unreachable("image_gather4_lz_2d_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4: {
    llvm_unreachable("mfma_scale_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_tied_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_tied_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x16_f16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_f16_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu4_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x16_iu8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w64_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12: {
    llvm_unreachable("wmma_* gfx12 NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x32_f16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x32_bf16_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x32_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x64_iu4_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64: {
    llvm_unreachable("swmmac_* gfx12 NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x4_f32:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x32_f16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x32_f16:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_bf16f32_16x16x32_bf16:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f16_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_wmma_i32_16x16x64_iu8:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_f32_32x16x128_f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale_f32_32x16x128_f4:
  case AMDGPU::BI__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4: {
    llvm_unreachable("wmma_scale_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_f16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x64_f16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_bf16f32_16x16x64_bf16:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f32_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_fp8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_fp8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_f16_16x16x128_bf8_bf8:
  case AMDGPU::BI__builtin_amdgcn_swmmac_i32_16x16x128_iu8: {
    llvm_unreachable("swmmac_* NYI");
  }
  // amdgcn workgroup size
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_x:
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_y:
  case AMDGPU::BI__builtin_amdgcn_workgroup_size_z: {
    llvm_unreachable("workgroup_size_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_grid_size_x:
  case AMDGPU::BI__builtin_amdgcn_grid_size_y:
  case AMDGPU::BI__builtin_amdgcn_grid_size_z: {
    llvm_unreachable("grid_size_* NYI");
  }
  case AMDGPU::BI__builtin_r600_recipsqrt_ieee:
  case AMDGPU::BI__builtin_r600_recipsqrt_ieeef: {
    llvm_unreachable("recipsqrt_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_alignbit: {
    llvm_unreachable("alignbit_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_fence: {
    llvm_unreachable("fence_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_atomic_inc32:
  case AMDGPU::BI__builtin_amdgcn_atomic_inc64:
  case AMDGPU::BI__builtin_amdgcn_atomic_dec32:
  case AMDGPU::BI__builtin_amdgcn_atomic_dec64:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_ds_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_ds_faddf:
  case AMDGPU::BI__builtin_amdgcn_ds_fminf:
  case AMDGPU::BI__builtin_amdgcn_ds_fmaxf:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2f16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fadd_v2bf16:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fmin_f64:
  case AMDGPU::BI__builtin_amdgcn_global_atomic_fmax_f64:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmin_f64:
  case AMDGPU::BI__builtin_amdgcn_flat_atomic_fmax_f64: {
    llvm_unreachable("atomic_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_s_sendmsg_rtn:
  case AMDGPU::BI__builtin_amdgcn_s_sendmsg_rtnl: {
    llvm_unreachable("s_sendmsg_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_permlane16_swap:
  case AMDGPU::BI__builtin_amdgcn_permlane32_swap: {
    llvm_unreachable("permlane_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_bitop3_b32:
  case AMDGPU::BI__builtin_amdgcn_bitop3_b16: {
    llvm_unreachable("bitop3_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_make_buffer_rsrc: {
    llvm_unreachable("make_buffer_rsrc_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b8:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b16:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b32:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b64:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b96:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_store_b128: {
    llvm_unreachable("raw_buffer_store_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b8:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b16:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b32:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b64:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b96:
  case AMDGPU::BI__builtin_amdgcn_raw_buffer_load_b128: {
    llvm_unreachable("raw_buffer_load_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_add_i32: {
    llvm_unreachable("raw_ptr_buffer_atomic_add_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16: {
    llvm_unreachable("raw_ptr_buffer_atomic_fadd_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64: {
    llvm_unreachable("raw_ptr_buffer_atomic_fmin_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32:
  case AMDGPU::BI__builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64: {
    llvm_unreachable("raw_ptr_buffer_atomic_fmax_* NYI");
  }
  case AMDGPU::BI__builtin_amdgcn_s_prefetch_data: {
    llvm_unreachable("s_prefetch_data_* NYI");
  }
  case Builtin::BIlogbf:
  case Builtin::BI__builtin_logbf:
    return emitLogbBuiltin(*this, expr, /*IsFloat=*/true);
  case Builtin::BIlogb:
  case Builtin::BI__builtin_logb:
    return emitLogbBuiltin(*this, expr, /*IsFloat=*/false);
  case Builtin::BIscalbnf:
  case Builtin::BI__builtin_scalbnf:
  case Builtin::BIscalbn:
  case Builtin::BI__builtin_scalbn: {
    return emitBinaryExpMaybeConstrainedFPBuiltin(
        *this, expr, "llvm.ldexp", "llvm.experimental.constrained.ldexp");
  }
  default:
    return nullptr;
  }
}
