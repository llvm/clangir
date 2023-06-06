//===---- CIRGenBuiltin.cpp - Emit CIR for builtins -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as CIR or a function call to be
// later resolved.
//
//===----------------------------------------------------------------------===//

#include "CIRGenCXXABI.h"
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "UnimplementedFeatureGuarding.h"

// TODO(cir): we shouldn't need this but we currently reuse intrinsic IDs for
// convenience.
#include "llvm/IR/Intrinsics.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;
using namespace llvm;

static RValue buildLibraryCall(CIRGenFunction &CGF, const FunctionDecl *FD,
                               const CallExpr *E,
                               mlir::Operation *calleeValue) {
  auto callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(FD));
  return CGF.buildCall(E->getCallee()->getType(), callee, E, ReturnValueSlot());
}

RValue CIRGenFunction::buildBuiltinExpr(const GlobalDecl GD, unsigned BuiltinID,
                                        const CallExpr *E,
                                        ReturnValueSlot ReturnValue) {
  const FunctionDecl *FD = GD.getDecl()->getAsFunction();

  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult Result;
  if (E->isPRValue() && E->EvaluateAsRValue(Result, CGM.getASTContext()) &&
      !Result.hasSideEffects()) {
    llvm_unreachable("NYI");
  }

  // If current long-double semantics is IEEE 128-bit, replace math builtins
  // of long-double with f128 equivalent.
  // TODO: This mutation should also be applied to other targets other than PPC,
  // after backend supports IEEE 128-bit style libcalls.
  if (getTarget().getTriple().isPPC64() &&
      &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad())
    llvm_unreachable("NYI");

  // If the builtin has been declared explicitly with an assembler label,
  // disable the specialized emitting below. Ideally we should communicate the
  // rename in IR, or at least avoid generating the intrinsic calls that are
  // likely to get lowered to the renamed library functions.
  const unsigned BuiltinIDIfNoAsmLabel =
      FD->hasAttr<AsmLabelAttr>() ? 0 : BuiltinID;

  // There are LLVM math intrinsics/instructions corresponding to math library
  // functions except the LLVM op will never set errno while the math library
  // might. Also, math builtins have the same semantics as their math library
  // twins. Thus, we can transform math library and builtin calls to their
  // LLVM counterparts if the call is marked 'const' (known to never set errno).
  // In case FP exceptions are enabled, the experimental versions of the
  // intrinsics model those.
  bool ConstWithoutErrnoAndExceptions =
      getContext().BuiltinInfo.isConstWithoutErrnoAndExceptions(BuiltinID);
  bool ConstWithoutExceptions =
      getContext().BuiltinInfo.isConstWithoutExceptions(BuiltinID);
  if (FD->hasAttr<ConstAttr>() ||
      ((ConstWithoutErrnoAndExceptions || ConstWithoutExceptions) &&
       (!ConstWithoutErrnoAndExceptions || (!getLangOpts().MathErrno)))) {
    switch (BuiltinIDIfNoAsmLabel) {
    case Builtin::BIceil:
    case Builtin::BIceilf:
    case Builtin::BIceill:
    case Builtin::BI__builtin_ceil:
    case Builtin::BI__builtin_ceilf:
    case Builtin::BI__builtin_ceilf16:
    case Builtin::BI__builtin_ceill:
    case Builtin::BI__builtin_ceilf128:
      llvm_unreachable("NYI");

    case Builtin::BIcopysign:
    case Builtin::BIcopysignf:
    case Builtin::BIcopysignl:
    case Builtin::BI__builtin_copysign:
    case Builtin::BI__builtin_copysignf:
    case Builtin::BI__builtin_copysignf16:
    case Builtin::BI__builtin_copysignl:
    case Builtin::BI__builtin_copysignf128:
      llvm_unreachable("NYI");

    case Builtin::BIcos:
    case Builtin::BIcosf:
    case Builtin::BIcosl:
    case Builtin::BI__builtin_cos:
    case Builtin::BI__builtin_cosf:
    case Builtin::BI__builtin_cosf16:
    case Builtin::BI__builtin_cosl:
    case Builtin::BI__builtin_cosf128:
      llvm_unreachable("NYI");

    case Builtin::BIexp:
    case Builtin::BIexpf:
    case Builtin::BIexpl:
    case Builtin::BI__builtin_exp:
    case Builtin::BI__builtin_expf:
    case Builtin::BI__builtin_expf16:
    case Builtin::BI__builtin_expl:
    case Builtin::BI__builtin_expf128:
      llvm_unreachable("NYI");

    case Builtin::BIexp2:
    case Builtin::BIexp2f:
    case Builtin::BIexp2l:
    case Builtin::BI__builtin_exp2:
    case Builtin::BI__builtin_exp2f:
    case Builtin::BI__builtin_exp2f16:
    case Builtin::BI__builtin_exp2l:
    case Builtin::BI__builtin_exp2f128:
      llvm_unreachable("NYI");

    case Builtin::BIfabs:
    case Builtin::BIfabsf:
    case Builtin::BIfabsl:
    case Builtin::BI__builtin_fabs:
    case Builtin::BI__builtin_fabsf:
    case Builtin::BI__builtin_fabsf16:
    case Builtin::BI__builtin_fabsl:
    case Builtin::BI__builtin_fabsf128:
      llvm_unreachable("NYI");

    case Builtin::BIfloor:
    case Builtin::BIfloorf:
    case Builtin::BIfloorl:
    case Builtin::BI__builtin_floor:
    case Builtin::BI__builtin_floorf:
    case Builtin::BI__builtin_floorf16:
    case Builtin::BI__builtin_floorl:
    case Builtin::BI__builtin_floorf128:
      llvm_unreachable("NYI");

    case Builtin::BIfma:
    case Builtin::BIfmaf:
    case Builtin::BIfmal:
    case Builtin::BI__builtin_fma:
    case Builtin::BI__builtin_fmaf:
    case Builtin::BI__builtin_fmaf16:
    case Builtin::BI__builtin_fmal:
    case Builtin::BI__builtin_fmaf128:
      llvm_unreachable("NYI");

    case Builtin::BIfmax:
    case Builtin::BIfmaxf:
    case Builtin::BIfmaxl:
    case Builtin::BI__builtin_fmax:
    case Builtin::BI__builtin_fmaxf:
    case Builtin::BI__builtin_fmaxf16:
    case Builtin::BI__builtin_fmaxl:
    case Builtin::BI__builtin_fmaxf128:
      llvm_unreachable("NYI");

    case Builtin::BIfmin:
    case Builtin::BIfminf:
    case Builtin::BIfminl:
    case Builtin::BI__builtin_fmin:
    case Builtin::BI__builtin_fminf:
    case Builtin::BI__builtin_fminf16:
    case Builtin::BI__builtin_fminl:
    case Builtin::BI__builtin_fminf128:
      llvm_unreachable("NYI");

    // fmod() is a special-case. It maps to the frem instruction rather than an
    // LLVM intrinsic.
    case Builtin::BIfmod:
    case Builtin::BIfmodf:
    case Builtin::BIfmodl:
    case Builtin::BI__builtin_fmod:
    case Builtin::BI__builtin_fmodf:
    case Builtin::BI__builtin_fmodf16:
    case Builtin::BI__builtin_fmodl:
    case Builtin::BI__builtin_fmodf128: {
      llvm_unreachable("NYI");
    }

    case Builtin::BIlog:
    case Builtin::BIlogf:
    case Builtin::BIlogl:
    case Builtin::BI__builtin_log:
    case Builtin::BI__builtin_logf:
    case Builtin::BI__builtin_logf16:
    case Builtin::BI__builtin_logl:
    case Builtin::BI__builtin_logf128:
      llvm_unreachable("NYI");

    case Builtin::BIlog10:
    case Builtin::BIlog10f:
    case Builtin::BIlog10l:
    case Builtin::BI__builtin_log10:
    case Builtin::BI__builtin_log10f:
    case Builtin::BI__builtin_log10f16:
    case Builtin::BI__builtin_log10l:
    case Builtin::BI__builtin_log10f128:
      llvm_unreachable("NYI");

    case Builtin::BIlog2:
    case Builtin::BIlog2f:
    case Builtin::BIlog2l:
    case Builtin::BI__builtin_log2:
    case Builtin::BI__builtin_log2f:
    case Builtin::BI__builtin_log2f16:
    case Builtin::BI__builtin_log2l:
    case Builtin::BI__builtin_log2f128:
      llvm_unreachable("NYI");

    case Builtin::BInearbyint:
    case Builtin::BInearbyintf:
    case Builtin::BInearbyintl:
    case Builtin::BI__builtin_nearbyint:
    case Builtin::BI__builtin_nearbyintf:
    case Builtin::BI__builtin_nearbyintl:
    case Builtin::BI__builtin_nearbyintf128:
      llvm_unreachable("NYI");

    case Builtin::BIpow:
    case Builtin::BIpowf:
    case Builtin::BIpowl:
    case Builtin::BI__builtin_pow:
    case Builtin::BI__builtin_powf:
    case Builtin::BI__builtin_powf16:
    case Builtin::BI__builtin_powl:
    case Builtin::BI__builtin_powf128:
      llvm_unreachable("NYI");

    case Builtin::BIrint:
    case Builtin::BIrintf:
    case Builtin::BIrintl:
    case Builtin::BI__builtin_rint:
    case Builtin::BI__builtin_rintf:
    case Builtin::BI__builtin_rintf16:
    case Builtin::BI__builtin_rintl:
    case Builtin::BI__builtin_rintf128:
      llvm_unreachable("NYI");

    case Builtin::BIround:
    case Builtin::BIroundf:
    case Builtin::BIroundl:
    case Builtin::BI__builtin_round:
    case Builtin::BI__builtin_roundf:
    case Builtin::BI__builtin_roundf16:
    case Builtin::BI__builtin_roundl:
    case Builtin::BI__builtin_roundf128:
      llvm_unreachable("NYI");

    case Builtin::BIsin:
    case Builtin::BIsinf:
    case Builtin::BIsinl:
    case Builtin::BI__builtin_sin:
    case Builtin::BI__builtin_sinf:
    case Builtin::BI__builtin_sinf16:
    case Builtin::BI__builtin_sinl:
    case Builtin::BI__builtin_sinf128:
      llvm_unreachable("NYI");

    case Builtin::BIsqrt:
    case Builtin::BIsqrtf:
    case Builtin::BIsqrtl:
    case Builtin::BI__builtin_sqrt:
    case Builtin::BI__builtin_sqrtf:
    case Builtin::BI__builtin_sqrtf16:
    case Builtin::BI__builtin_sqrtl:
    case Builtin::BI__builtin_sqrtf128:
      llvm_unreachable("NYI");

    case Builtin::BItrunc:
    case Builtin::BItruncf:
    case Builtin::BItruncl:
    case Builtin::BI__builtin_trunc:
    case Builtin::BI__builtin_truncf:
    case Builtin::BI__builtin_truncf16:
    case Builtin::BI__builtin_truncl:
    case Builtin::BI__builtin_truncf128:
      llvm_unreachable("NYI");

    case Builtin::BIlround:
    case Builtin::BIlroundf:
    case Builtin::BIlroundl:
    case Builtin::BI__builtin_lround:
    case Builtin::BI__builtin_lroundf:
    case Builtin::BI__builtin_lroundl:
    case Builtin::BI__builtin_lroundf128:
      llvm_unreachable("NYI");

    case Builtin::BIllround:
    case Builtin::BIllroundf:
    case Builtin::BIllroundl:
    case Builtin::BI__builtin_llround:
    case Builtin::BI__builtin_llroundf:
    case Builtin::BI__builtin_llroundl:
    case Builtin::BI__builtin_llroundf128:
      llvm_unreachable("NYI");

    case Builtin::BIlrint:
    case Builtin::BIlrintf:
    case Builtin::BIlrintl:
    case Builtin::BI__builtin_lrint:
    case Builtin::BI__builtin_lrintf:
    case Builtin::BI__builtin_lrintl:
    case Builtin::BI__builtin_lrintf128:
      llvm_unreachable("NYI");

    case Builtin::BIllrint:
    case Builtin::BIllrintf:
    case Builtin::BIllrintl:
    case Builtin::BI__builtin_llrint:
    case Builtin::BI__builtin_llrintf:
    case Builtin::BI__builtin_llrintl:
    case Builtin::BI__builtin_llrintf128:
      llvm_unreachable("NYI");

    default:
      break;
    }
  }

  switch (BuiltinIDIfNoAsmLabel) {
  default:
    llvm_unreachable("NYI");
    break;

  case Builtin::BIprintf:
    if (getTarget().getTriple().isNVPTX() ||
        getTarget().getTriple().isAMDGCN()) {
      llvm_unreachable("NYI");
    }
    break;

  // C++ std:: builtins.
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIforward:
  case Builtin::BIas_const:
    return RValue::get(buildLValue(E->getArg(0)).getPointer());
  case Builtin::BI__GetExceptionInfo: {
    llvm_unreachable("NYI");
  }

  case Builtin::BI__fastfail:
    llvm_unreachable("NYI");

  case Builtin::BI__builtin_coro_id:
  case Builtin::BI__builtin_coro_promise:
  case Builtin::BI__builtin_coro_resume:
  case Builtin::BI__builtin_coro_noop:
  case Builtin::BI__builtin_coro_destroy:
  case Builtin::BI__builtin_coro_done:
  case Builtin::BI__builtin_coro_alloc:
  case Builtin::BI__builtin_coro_begin:
  case Builtin::BI__builtin_coro_end:
  case Builtin::BI__builtin_coro_suspend:
  case Builtin::BI__builtin_coro_align:
    llvm_unreachable("NYI");

  case Builtin::BI__builtin_coro_frame: {
    return buildCoroutineFrame();
  }
  case Builtin::BI__builtin_coro_free:
  case Builtin::BI__builtin_coro_size: {
    GlobalDecl gd{FD};
    mlir::Type ty = CGM.getTypes().GetFunctionType(
        CGM.getTypes().arrangeGlobalDeclaration(GD));
    const auto *ND = cast<NamedDecl>(GD.getDecl());
    auto fnOp =
        CGM.GetOrCreateCIRFunction(ND->getName(), ty, gd, /*ForVTable=*/false,
                                   /*DontDefer=*/false);
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
    return buildCall(E->getCallee()->getType(), CIRGenCallee::forDirect(fnOp),
                     E, ReturnValue);
  }
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(BuiltinID))
    llvm_unreachable("NYI");

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(BuiltinID))
    return buildLibraryCall(*this, FD, E,
                            buildScalarExpr(E->getCallee()).getDefiningOp());

  // Check that a call to a target specific builtin has the correct target
  // features.
  // This is down here to avoid non-target specific builtins, however, if
  // generic builtins start to require generic target features then we
  // can move this up to the beginning of the function.
  //   checkTargetFeatures(E, FD);

  if (unsigned VectorWidth =
          getContext().BuiltinInfo.getRequiredVectorWidth(BuiltinID))
    llvm_unreachable("NYI");

  // See if we have a target specific intrinsic.
  auto Name = getContext().BuiltinInfo.getName(BuiltinID).str();
  Intrinsic::ID IntrinsicID = Intrinsic::not_intrinsic;
  StringRef Prefix =
      llvm::Triple::getArchTypePrefix(getTarget().getTriple().getArch());
  if (!Prefix.empty()) {
    IntrinsicID = Intrinsic::getIntrinsicForClangBuiltin(Prefix.data(), Name);
    // NOTE we don't need to perform a compatibility flag check here since the
    // intrinsics are declared in Builtins*.def via LANGBUILTIN which filter the
    // MS builtins via ALL_MS_LANGUAGES and are filtered earlier.
    if (IntrinsicID == Intrinsic::not_intrinsic)
      IntrinsicID = Intrinsic::getIntrinsicForMSBuiltin(Prefix.data(), Name);
  }

  if (IntrinsicID != Intrinsic::not_intrinsic) {
    llvm_unreachable("NYI");
  }

  // Some target-specific builtins can have aggregate return values, e.g.
  // __builtin_arm_mve_vld2q_u32. So if the result is an aggregate, force
  // ReturnValue to be non-null, so that the target-specific emission code can
  // always just emit into it.
  TypeEvaluationKind EvalKind = getEvaluationKind(E->getType());
  if (EvalKind == TEK_Aggregate && ReturnValue.isNull()) {
    llvm_unreachable("NYI");
  }

  // Now see if we can emit a target-specific builtin.
  if (auto v = buildTargetBuiltinExpr(BuiltinID, E, ReturnValue)) {
    llvm_unreachable("NYI");
  }

  llvm_unreachable("NYI");
  //   ErrorUnsupported(E, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  return GetUndefRValue(E->getType());
}

static mlir::Value buildTargetArchBuiltinExpr(CIRGenFunction *CGF,
                                              unsigned BuiltinID,
                                              const CallExpr *E,
                                              ReturnValueSlot ReturnValue,
                                              llvm::Triple::ArchType Arch) {
  llvm_unreachable("NYI");
  return {};
}

mlir::Value
CIRGenFunction::buildTargetBuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                       ReturnValueSlot ReturnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(BuiltinID)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return buildTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(BuiltinID), E,
        ReturnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return buildTargetArchBuiltinExpr(this, BuiltinID, E, ReturnValue,
                                    getTarget().getTriple().getArch());
}
