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
  [[maybe_unused]] auto getIntrinsic = [&](const char *name) {
    mlir::Type intTy = cir::IntType::get(&getMLIRContext(), 32, false);
    return builder
        .create<cir::LLVMIntrinsicCallOp>(getLoc(expr->getExprLoc()),
                                          builder.getStringAttr(name), intTy)
        .getResult();
  };
  switch (builtinId) {
  case NVPTX::BI__nvvm_atom_add_gen_i:
  case NVPTX::BI__nvvm_atom_add_gen_l:
  case NVPTX::BI__nvvm_atom_add_gen_ll:
    llvm_unreachable("atom_add_gen_* NYI");

  case NVPTX::BI__nvvm_atom_sub_gen_i:
  case NVPTX::BI__nvvm_atom_sub_gen_l:
  case NVPTX::BI__nvvm_atom_sub_gen_ll:
    llvm_unreachable("atom_sub_gen_* NYI");

  case NVPTX::BI__nvvm_atom_and_gen_i:
  case NVPTX::BI__nvvm_atom_and_gen_l:
  case NVPTX::BI__nvvm_atom_and_gen_ll:
    llvm_unreachable("atom_and_gen_ll NYI");

  case NVPTX::BI__nvvm_atom_or_gen_i:
  case NVPTX::BI__nvvm_atom_or_gen_l:
  case NVPTX::BI__nvvm_atom_or_gen_ll:
    llvm_unreachable("atom_or_gen_* NYI");

  case NVPTX::BI__nvvm_atom_xor_gen_i:
  case NVPTX::BI__nvvm_atom_xor_gen_l:
  case NVPTX::BI__nvvm_atom_xor_gen_ll:
    llvm_unreachable("atom_xor_gen_* NYI");

  case NVPTX::BI__nvvm_atom_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_xchg_gen_ll:
    llvm_unreachable("atom_xchg_gen_* NYI");

  case NVPTX::BI__nvvm_atom_max_gen_i:
  case NVPTX::BI__nvvm_atom_max_gen_l:
  case NVPTX::BI__nvvm_atom_max_gen_ll:
    llvm_unreachable("atom_max_gen_* NYI");

  case NVPTX::BI__nvvm_atom_max_gen_ui:
  case NVPTX::BI__nvvm_atom_max_gen_ul:
  case NVPTX::BI__nvvm_atom_max_gen_ull:
    llvm_unreachable("atom_max_gen_* NYI");

  case NVPTX::BI__nvvm_atom_min_gen_i:
  case NVPTX::BI__nvvm_atom_min_gen_l:
  case NVPTX::BI__nvvm_atom_min_gen_ll:
    llvm_unreachable("atom_min_gen_* NYI");

  case NVPTX::BI__nvvm_atom_min_gen_ui:
  case NVPTX::BI__nvvm_atom_min_gen_ul:
  case NVPTX::BI__nvvm_atom_min_gen_ull:
    llvm_unreachable("atom_min_gen_* NYI");

  case NVPTX::BI__nvvm_atom_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cas_gen_ll:
    // __nvvm_atom_cas_gen_* should return the old value rather than the
    // success flag.
    llvm_unreachable("atom_cas_gen_* NYI");

  case NVPTX::BI__nvvm_atom_add_gen_f:
  case NVPTX::BI__nvvm_atom_add_gen_d:
    llvm_unreachable("atom_add_gen_f/d NYI");

  case NVPTX::BI__nvvm_atom_inc_gen_ui:
    llvm_unreachable("atom_inc_gen_ui NYI");

  case NVPTX::BI__nvvm_atom_dec_gen_ui:
    llvm_unreachable("atom_dec_gen_ui NYI");

  case NVPTX::BI__nvvm_ldg_c:
  case NVPTX::BI__nvvm_ldg_sc:
  case NVPTX::BI__nvvm_ldg_c2:
  case NVPTX::BI__nvvm_ldg_sc2:
  case NVPTX::BI__nvvm_ldg_c4:
  case NVPTX::BI__nvvm_ldg_sc4:
  case NVPTX::BI__nvvm_ldg_s:
  case NVPTX::BI__nvvm_ldg_s2:
  case NVPTX::BI__nvvm_ldg_s4:
  case NVPTX::BI__nvvm_ldg_i:
  case NVPTX::BI__nvvm_ldg_i2:
  case NVPTX::BI__nvvm_ldg_i4:
  case NVPTX::BI__nvvm_ldg_l:
  case NVPTX::BI__nvvm_ldg_l2:
  case NVPTX::BI__nvvm_ldg_ll:
  case NVPTX::BI__nvvm_ldg_ll2:
  case NVPTX::BI__nvvm_ldg_uc:
  case NVPTX::BI__nvvm_ldg_uc2:
  case NVPTX::BI__nvvm_ldg_uc4:
  case NVPTX::BI__nvvm_ldg_us:
  case NVPTX::BI__nvvm_ldg_us2:
  case NVPTX::BI__nvvm_ldg_us4:
  case NVPTX::BI__nvvm_ldg_ui:
  case NVPTX::BI__nvvm_ldg_ui2:
  case NVPTX::BI__nvvm_ldg_ui4:
  case NVPTX::BI__nvvm_ldg_ul:
  case NVPTX::BI__nvvm_ldg_ul2:
  case NVPTX::BI__nvvm_ldg_ull:
  case NVPTX::BI__nvvm_ldg_ull2:
  case NVPTX::BI__nvvm_ldg_f:
  case NVPTX::BI__nvvm_ldg_f2:
  case NVPTX::BI__nvvm_ldg_f4:
  case NVPTX::BI__nvvm_ldg_d:
  case NVPTX::BI__nvvm_ldg_d2:
    llvm_unreachable("ldg_* NYI");

  case NVPTX::BI__nvvm_ldu_c:
  case NVPTX::BI__nvvm_ldu_sc:
  case NVPTX::BI__nvvm_ldu_c2:
  case NVPTX::BI__nvvm_ldu_sc2:
  case NVPTX::BI__nvvm_ldu_c4:
  case NVPTX::BI__nvvm_ldu_sc4:
  case NVPTX::BI__nvvm_ldu_s:
  case NVPTX::BI__nvvm_ldu_s2:
  case NVPTX::BI__nvvm_ldu_s4:
  case NVPTX::BI__nvvm_ldu_i:
  case NVPTX::BI__nvvm_ldu_i2:
  case NVPTX::BI__nvvm_ldu_i4:
  case NVPTX::BI__nvvm_ldu_l:
  case NVPTX::BI__nvvm_ldu_l2:
  case NVPTX::BI__nvvm_ldu_ll:
  case NVPTX::BI__nvvm_ldu_ll2:
  case NVPTX::BI__nvvm_ldu_uc:
  case NVPTX::BI__nvvm_ldu_uc2:
  case NVPTX::BI__nvvm_ldu_uc4:
  case NVPTX::BI__nvvm_ldu_us:
  case NVPTX::BI__nvvm_ldu_us2:
  case NVPTX::BI__nvvm_ldu_us4:
  case NVPTX::BI__nvvm_ldu_ui:
  case NVPTX::BI__nvvm_ldu_ui2:
  case NVPTX::BI__nvvm_ldu_ui4:
  case NVPTX::BI__nvvm_ldu_ul:
  case NVPTX::BI__nvvm_ldu_ul2:
  case NVPTX::BI__nvvm_ldu_ull:
  case NVPTX::BI__nvvm_ldu_ull2:
    llvm_unreachable("ldu_* NYI");
  case NVPTX::BI__nvvm_ldu_f:
  case NVPTX::BI__nvvm_ldu_f2:
  case NVPTX::BI__nvvm_ldu_f4:
  case NVPTX::BI__nvvm_ldu_d:
  case NVPTX::BI__nvvm_ldu_d2:
    llvm_unreachable("ldu_* NYI");

  case NVPTX::BI__nvvm_atom_cta_add_gen_i:
  case NVPTX::BI__nvvm_atom_cta_add_gen_l:
  case NVPTX::BI__nvvm_atom_cta_add_gen_ll:
    llvm_unreachable("atom_cta_add_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_add_gen_i:
  case NVPTX::BI__nvvm_atom_sys_add_gen_l:
  case NVPTX::BI__nvvm_atom_sys_add_gen_ll:
    llvm_unreachable("atom_sys_add_gen_* NYI");
  case NVPTX::BI__nvvm_atom_cta_add_gen_f:
  case NVPTX::BI__nvvm_atom_cta_add_gen_d:
    llvm_unreachable("atom_cta_add_gen_f/d NYI");
  case NVPTX::BI__nvvm_atom_sys_add_gen_f:
  case NVPTX::BI__nvvm_atom_sys_add_gen_d:
    llvm_unreachable("atom_sys_add_gen_f/d NYI");
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xchg_gen_ll:
    llvm_unreachable("atom_cta_xchg_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xchg_gen_ll:
    llvm_unreachable("atom_sys_xchg_gen_* NYI");
  case NVPTX::BI__nvvm_atom_cta_max_gen_i:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_max_gen_l:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_max_gen_ull:
    llvm_unreachable("atom_cta_max_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_max_gen_i:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_max_gen_l:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_max_gen_ull:
    llvm_unreachable("atom_sys_max_gen_* NYI");
  case NVPTX::BI__nvvm_atom_cta_min_gen_i:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ui:
  case NVPTX::BI__nvvm_atom_cta_min_gen_l:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ul:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ll:
  case NVPTX::BI__nvvm_atom_cta_min_gen_ull:
    llvm_unreachable("atom_cta_min_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_min_gen_i:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ui:
  case NVPTX::BI__nvvm_atom_sys_min_gen_l:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ul:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ll:
  case NVPTX::BI__nvvm_atom_sys_min_gen_ull:
    llvm_unreachable("atom_sys_min_gen_* NYI");
  case NVPTX::BI__nvvm_atom_cta_inc_gen_ui:
    llvm_unreachable("atom_cta_inc_gen_ui NYI");
  case NVPTX::BI__nvvm_atom_cta_dec_gen_ui:
    llvm_unreachable("atom_cta_dec_gen_ui NYI");
  case NVPTX::BI__nvvm_atom_sys_inc_gen_ui:
    llvm_unreachable("atom_sys_inc_gen_ui NYI");
  case NVPTX::BI__nvvm_atom_sys_dec_gen_ui:
    llvm_unreachable("atom_sys_dec_gen_ui NYI");
  case NVPTX::BI__nvvm_atom_cta_and_gen_i:
  case NVPTX::BI__nvvm_atom_cta_and_gen_l:
  case NVPTX::BI__nvvm_atom_cta_and_gen_ll:
    llvm_unreachable("atom_cta_and_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_and_gen_i:
  case NVPTX::BI__nvvm_atom_sys_and_gen_l:
  case NVPTX::BI__nvvm_atom_sys_and_gen_ll:
    llvm_unreachable("atom_sys_and_gen_* NYI");
  case NVPTX::BI__nvvm_atom_cta_or_gen_i:
  case NVPTX::BI__nvvm_atom_cta_or_gen_l:
  case NVPTX::BI__nvvm_atom_cta_or_gen_ll:
    llvm_unreachable("atom_cta_or_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_or_gen_i:
  case NVPTX::BI__nvvm_atom_sys_or_gen_l:
  case NVPTX::BI__nvvm_atom_sys_or_gen_ll:
    llvm_unreachable("atom_sys_or_gen_* NYI");
  case NVPTX::BI__nvvm_atom_cta_xor_gen_i:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_l:
  case NVPTX::BI__nvvm_atom_cta_xor_gen_ll:
    llvm_unreachable("atom_cta_xor_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_xor_gen_i:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_l:
  case NVPTX::BI__nvvm_atom_sys_xor_gen_ll:
    llvm_unreachable("atom_sys_xor_gen_* NYI");
  case NVPTX::BI__nvvm_atom_cta_cas_gen_us:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_i:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_l:
  case NVPTX::BI__nvvm_atom_cta_cas_gen_ll:
    llvm_unreachable("atom_cta_cas_gen_* NYI");
  case NVPTX::BI__nvvm_atom_sys_cas_gen_us:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_i:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_l:
  case NVPTX::BI__nvvm_atom_sys_cas_gen_ll:
    llvm_unreachable("atom_sys_cas_gen_* NYI");
  case NVPTX::BI__nvvm_match_all_sync_i32p:
  case NVPTX::BI__nvvm_match_all_sync_i64p:
    llvm_unreachable("match_all_sync_* NYI");
  // FP MMA loads
  case NVPTX::BI__hmma_m16n16k16_ld_a:
  case NVPTX::BI__hmma_m16n16k16_ld_b:
  case NVPTX::BI__hmma_m16n16k16_ld_c_f16:
  case NVPTX::BI__hmma_m16n16k16_ld_c_f32:
  case NVPTX::BI__hmma_m32n8k16_ld_a:
  case NVPTX::BI__hmma_m32n8k16_ld_b:
  case NVPTX::BI__hmma_m32n8k16_ld_c_f16:
  case NVPTX::BI__hmma_m32n8k16_ld_c_f32:
  case NVPTX::BI__hmma_m8n32k16_ld_a:
  case NVPTX::BI__hmma_m8n32k16_ld_b:
  case NVPTX::BI__hmma_m8n32k16_ld_c_f16:
  case NVPTX::BI__hmma_m8n32k16_ld_c_f32:
  // Integer MMA loads.
  case NVPTX::BI__imma_m16n16k16_ld_a_s8:
  case NVPTX::BI__imma_m16n16k16_ld_a_u8:
  case NVPTX::BI__imma_m16n16k16_ld_b_s8:
  case NVPTX::BI__imma_m16n16k16_ld_b_u8:
  case NVPTX::BI__imma_m16n16k16_ld_c:
  case NVPTX::BI__imma_m32n8k16_ld_a_s8:
  case NVPTX::BI__imma_m32n8k16_ld_a_u8:
  case NVPTX::BI__imma_m32n8k16_ld_b_s8:
  case NVPTX::BI__imma_m32n8k16_ld_b_u8:
  case NVPTX::BI__imma_m32n8k16_ld_c:
  case NVPTX::BI__imma_m8n32k16_ld_a_s8:
  case NVPTX::BI__imma_m8n32k16_ld_a_u8:
  case NVPTX::BI__imma_m8n32k16_ld_b_s8:
  case NVPTX::BI__imma_m8n32k16_ld_b_u8:
  case NVPTX::BI__imma_m8n32k16_ld_c:
  // Sub-integer MMA loads.
  case NVPTX::BI__imma_m8n8k32_ld_a_s4:
  case NVPTX::BI__imma_m8n8k32_ld_a_u4:
  case NVPTX::BI__imma_m8n8k32_ld_b_s4:
  case NVPTX::BI__imma_m8n8k32_ld_b_u4:
  case NVPTX::BI__imma_m8n8k32_ld_c:
  case NVPTX::BI__bmma_m8n8k128_ld_a_b1:
  case NVPTX::BI__bmma_m8n8k128_ld_b_b1:
  case NVPTX::BI__bmma_m8n8k128_ld_c:
  // Double MMA loads.
  case NVPTX::BI__dmma_m8n8k4_ld_a:
  case NVPTX::BI__dmma_m8n8k4_ld_b:
  case NVPTX::BI__dmma_m8n8k4_ld_c:
  // Alternate float MMA loads.
  case NVPTX::BI__mma_bf16_m16n16k16_ld_a:
  case NVPTX::BI__mma_bf16_m16n16k16_ld_b:
  case NVPTX::BI__mma_bf16_m8n32k16_ld_a:
  case NVPTX::BI__mma_bf16_m8n32k16_ld_b:
  case NVPTX::BI__mma_bf16_m32n8k16_ld_a:
  case NVPTX::BI__mma_bf16_m32n8k16_ld_b:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_a:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_b:
  case NVPTX::BI__mma_tf32_m16n16k8_ld_c:
    llvm_unreachable("MMA load NYI");

  case NVPTX::BI__hmma_m16n16k16_st_c_f16:
  case NVPTX::BI__hmma_m16n16k16_st_c_f32:
  case NVPTX::BI__hmma_m32n8k16_st_c_f16:
  case NVPTX::BI__hmma_m32n8k16_st_c_f32:
  case NVPTX::BI__hmma_m8n32k16_st_c_f16:
  case NVPTX::BI__hmma_m8n32k16_st_c_f32:
  case NVPTX::BI__imma_m16n16k16_st_c_i32:
  case NVPTX::BI__imma_m32n8k16_st_c_i32:
  case NVPTX::BI__imma_m8n32k16_st_c_i32:
  case NVPTX::BI__imma_m8n8k32_st_c_i32:
  case NVPTX::BI__bmma_m8n8k128_st_c_i32:
  case NVPTX::BI__dmma_m8n8k4_st_c_f64:
  case NVPTX::BI__mma_m16n16k8_st_c_f32:
    llvm_unreachable("MMA store NYI");

  // BI__hmma_m16n16k16_mma_<Dtype><CType>(d, a, b, c, layout, satf) -->
  // Intrinsic::nvvm_wmma_m16n16k16_mma_sync<layout A,B><DType><CType><Satf>
  case NVPTX::BI__hmma_m16n16k16_mma_f16f16:
  case NVPTX::BI__hmma_m16n16k16_mma_f32f16:
  case NVPTX::BI__hmma_m16n16k16_mma_f32f32:
  case NVPTX::BI__hmma_m16n16k16_mma_f16f32:
  case NVPTX::BI__hmma_m32n8k16_mma_f16f16:
  case NVPTX::BI__hmma_m32n8k16_mma_f32f16:
  case NVPTX::BI__hmma_m32n8k16_mma_f32f32:
  case NVPTX::BI__hmma_m32n8k16_mma_f16f32:
  case NVPTX::BI__hmma_m8n32k16_mma_f16f16:
  case NVPTX::BI__hmma_m8n32k16_mma_f32f16:
  case NVPTX::BI__hmma_m8n32k16_mma_f32f32:
  case NVPTX::BI__hmma_m8n32k16_mma_f16f32:
  case NVPTX::BI__imma_m16n16k16_mma_s8:
  case NVPTX::BI__imma_m16n16k16_mma_u8:
  case NVPTX::BI__imma_m32n8k16_mma_s8:
  case NVPTX::BI__imma_m32n8k16_mma_u8:
  case NVPTX::BI__imma_m8n32k16_mma_s8:
  case NVPTX::BI__imma_m8n32k16_mma_u8:
  case NVPTX::BI__imma_m8n8k32_mma_s4:
  case NVPTX::BI__imma_m8n8k32_mma_u4:
  case NVPTX::BI__bmma_m8n8k128_mma_xor_popc_b1:
  case NVPTX::BI__bmma_m8n8k128_mma_and_popc_b1:
  case NVPTX::BI__dmma_m8n8k4_mma_f64:
  case NVPTX::BI__mma_bf16_m16n16k16_mma_f32:
  case NVPTX::BI__mma_bf16_m8n32k16_mma_f32:
  case NVPTX::BI__mma_bf16_m32n8k16_mma_f32:
  case NVPTX::BI__mma_tf32_m16n16k8_mma_f32:
    llvm_unreachable("MMA compute NYI");
  // The following builtins require half type support
  case NVPTX::BI__nvvm_ex2_approx_f16:
    llvm_unreachable("ex2_approx_f16 NYI");
  case NVPTX::BI__nvvm_ex2_approx_f16x2:
    llvm_unreachable("ex2_approx_f16x2 NYI");
  case NVPTX::BI__nvvm_ff2f16x2_rn:
    llvm_unreachable("ff2f16x2_rn NYI");
  case NVPTX::BI__nvvm_ff2f16x2_rn_relu:
    llvm_unreachable("ff2f16x2_rn_relu NYI");
  case NVPTX::BI__nvvm_ff2f16x2_rz:
    llvm_unreachable("ff2f16x2_rz NYI");
  case NVPTX::BI__nvvm_ff2f16x2_rz_relu:
    llvm_unreachable("ff2f16x2_rz_relu NYI");
  case NVPTX::BI__nvvm_fma_rn_f16:
    llvm_unreachable("fma_rn_f16 NYI");
  case NVPTX::BI__nvvm_fma_rn_f16x2:
    llvm_unreachable("fma_rn_f16x2 NYI");
  case NVPTX::BI__nvvm_fma_rn_ftz_f16:
    llvm_unreachable("fma_rn_ftz_f16 NYI");
  case NVPTX::BI__nvvm_fma_rn_ftz_f16x2:
    llvm_unreachable("fma_rn_ftz_f16x2 NYI");
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16:
    llvm_unreachable("fma_rn_ftz_relu_f16 NYI");
  case NVPTX::BI__nvvm_fma_rn_ftz_relu_f16x2:
    llvm_unreachable("fma_rn_ftz_relu_f16x2 NYI");
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16:
    llvm_unreachable("fma_rn_ftz_sat_f16 NYI");
  case NVPTX::BI__nvvm_fma_rn_ftz_sat_f16x2:
    llvm_unreachable("fma_rn_ftz_sat_f16x2 NYI");
  case NVPTX::BI__nvvm_fma_rn_relu_f16:
    llvm_unreachable("fma_rn_relu_f16 NYI");
  case NVPTX::BI__nvvm_fma_rn_relu_f16x2:
    llvm_unreachable("fma_rn_relu_f16x2 NYI");
  case NVPTX::BI__nvvm_fma_rn_sat_f16:
    llvm_unreachable("fma_rn_sat_f16 NYI");
  case NVPTX::BI__nvvm_fma_rn_sat_f16x2:
    llvm_unreachable("fma_rn_sat_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_f16:
    llvm_unreachable("fmax_f16 NYI");
  case NVPTX::BI__nvvm_fmax_f16x2:
    llvm_unreachable("fmax_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_f16:
    llvm_unreachable("fmax_ftz_f16 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_f16x2:
    llvm_unreachable("fmax_ftz_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16:
    llvm_unreachable("fmax_ftz_nan_f16 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_nan_f16x2:
    llvm_unreachable("fmax_ftz_nan_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16:
    llvm_unreachable("fmax_ftz_nan_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_nan_xorsign_abs_f16x2:
    llvm_unreachable("fmax_ftz_nan_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16:
    llvm_unreachable("fmax_ftz_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmax_ftz_xorsign_abs_f16x2:
    llvm_unreachable("fmax_ftz_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_nan_f16:
    llvm_unreachable("fmax_nan_f16 NYI");
  case NVPTX::BI__nvvm_fmax_nan_f16x2:
    llvm_unreachable("fmax_nan_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16:
    llvm_unreachable("fmax_nan_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmax_nan_xorsign_abs_f16x2:
    llvm_unreachable("fmax_nan_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16:
    llvm_unreachable("fmax_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmax_xorsign_abs_f16x2:
    llvm_unreachable("fmax_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_f16:
    llvm_unreachable("fmin_f16 NYI");
  case NVPTX::BI__nvvm_fmin_f16x2:
    llvm_unreachable("fmin_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_f16:
    llvm_unreachable("fmin_ftz_f16 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_f16x2:
    llvm_unreachable("fmin_ftz_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16:
    llvm_unreachable("fmin_ftz_nan_f16 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_nan_f16x2:
    llvm_unreachable("fmin_ftz_nan_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16:
    llvm_unreachable("fmin_ftz_nan_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_nan_xorsign_abs_f16x2:
    llvm_unreachable("fmin_ftz_nan_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16:
    llvm_unreachable("fmin_ftz_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmin_ftz_xorsign_abs_f16x2:
    llvm_unreachable("fmin_ftz_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_nan_f16:
    llvm_unreachable("fmin_nan_f16 NYI");
  case NVPTX::BI__nvvm_fmin_nan_f16x2:
    llvm_unreachable("fmin_nan_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16:
    llvm_unreachable("fmin_nan_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmin_nan_xorsign_abs_f16x2:
    llvm_unreachable("fmin_nan_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16:
    llvm_unreachable("fmin_xorsign_abs_f16 NYI");
  case NVPTX::BI__nvvm_fmin_xorsign_abs_f16x2:
    llvm_unreachable("fmin_xorsign_abs_f16x2 NYI");
  case NVPTX::BI__nvvm_fabs_f:
  case NVPTX::BI__nvvm_abs_bf16:
  case NVPTX::BI__nvvm_abs_bf16x2:
  case NVPTX::BI__nvvm_fabs_f16:
  case NVPTX::BI__nvvm_fabs_f16x2:
    llvm_unreachable("fabs_f16/ bf16 NYI");
  case NVPTX::BI__nvvm_fabs_ftz_f:
  case NVPTX::BI__nvvm_fabs_ftz_f16:
  case NVPTX::BI__nvvm_fabs_ftz_f16x2:
    llvm_unreachable("fabs_ftz_f16 NYI");
  case NVPTX::BI__nvvm_fabs_d:
    llvm_unreachable("fabs_d NYI");
  case NVPTX::BI__nvvm_ldg_h:
  case NVPTX::BI__nvvm_ldg_h2:
    llvm_unreachable("ldg_h NYI");
  case NVPTX::BI__nvvm_ldu_h:
  case NVPTX::BI__nvvm_ldu_h2:
    llvm_unreachable("ldu_h NYI");
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_4:
    llvm_unreachable("cp_async_ca_shared_global_4 NYI");
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_8:
    llvm_unreachable("cp_async_ca_shared_global_8 NYI");
  case NVPTX::BI__nvvm_cp_async_ca_shared_global_16:
    llvm_unreachable("cp_async_ca_shared_global_16 NYI");
  case NVPTX::BI__nvvm_cp_async_cg_shared_global_16:
    llvm_unreachable("cp_async_cg_shared_global_16 NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_x:
    llvm_unreachable("read_ptx_sreg_clusterid_x NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_y:
    llvm_unreachable("read_ptx_sreg_clusterid_y NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_z:
    llvm_unreachable("read_ptx_sreg_clusterid_z NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_clusterid_w:
    llvm_unreachable("read_ptx_sreg_clusterid_w NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_x:
    llvm_unreachable("read_ptx_sreg_nclusterid_x NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_y:
    llvm_unreachable("read_ptx_sreg_nclusterid_y NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_z:
    llvm_unreachable("read_ptx_sreg_nclusterid_z NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_nclusterid_w:
    llvm_unreachable("read_ptx_sreg_nclusterid_w NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_x:
    llvm_unreachable("read_ptx_sreg_cluster_ctaid_x NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_y:
    llvm_unreachable("read_ptx_sreg_cluster_ctaid_y NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_z:
    llvm_unreachable("read_ptx_sreg_cluster_ctaid_z NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctaid_w:
    llvm_unreachable("read_ptx_sreg_cluster_ctaid_w NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_x:
    llvm_unreachable("read_ptx_sreg_cluster_nctaid_x NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_y:
    llvm_unreachable("read_ptx_sreg_cluster_nctaid_y NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_z:
    llvm_unreachable("read_ptx_sreg_cluster_nctaid_z NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctaid_w:
    llvm_unreachable("read_ptx_sreg_cluster_nctaid_w NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_ctarank:
    llvm_unreachable("read_ptx_sreg_cluster_ctarank NYI");
  case NVPTX::BI__nvvm_read_ptx_sreg_cluster_nctarank:
    llvm_unreachable("read_ptx_sreg_cluster_nctarank NYI");
  case NVPTX::BI__nvvm_is_explicit_cluster:
    llvm_unreachable("is_explicit_cluster NYI");
  case NVPTX::BI__nvvm_isspacep_shared_cluster:
    llvm_unreachable("isspacep_shared_cluster NYI");
  case NVPTX::BI__nvvm_mapa:
    llvm_unreachable("mapa NYI");
  case NVPTX::BI__nvvm_mapa_shared_cluster:
    llvm_unreachable("mapa_shared_cluster NYI");
  case NVPTX::BI__nvvm_getctarank:
    llvm_unreachable("getctarank NYI");
  case NVPTX::BI__nvvm_getctarank_shared_cluster:
    llvm_unreachable("getctarank_shared_cluster NYI");
  case NVPTX::BI__nvvm_barrier_cluster_arrive:
    llvm_unreachable("barrier_cluster_arrive NYI");
  case NVPTX::BI__nvvm_barrier_cluster_arrive_relaxed:
    llvm_unreachable("barrier_cluster_arrive_relaxed NYI");
  case NVPTX::BI__nvvm_barrier_cluster_wait:
    llvm_unreachable("barrier_cluster_wait NYI");
  case NVPTX::BI__nvvm_fence_sc_cluster:
    llvm_unreachable("fence_sc_cluster NYI");
  case NVPTX::BI__nvvm_bar_sync:
    llvm_unreachable("bar_sync NYI");
  case NVPTX::BI__syncthreads:
    llvm_unreachable("syncthreads NYI");
  case NVPTX::BI__nvvm_barrier_sync:
    llvm_unreachable("barrier_sync NYI");
  case NVPTX::BI__nvvm_barrier_sync_cnt:
    llvm_unreachable("barrier_sync_cnt NYI");
  default:
    return nullptr;
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

  // We can directly store the arguments into a record, and the alignment
  // would automatically be correct. That's because vprintf does not
  // accept aggregates.
  mlir::Type allocaTy =
      cir::RecordType::get(&cgf.getMLIRContext(), argTypes, /*packed=*/false,
                           /*padded=*/false, cir::RecordType::Struct);
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
