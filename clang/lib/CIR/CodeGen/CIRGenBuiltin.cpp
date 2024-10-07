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
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"

// TODO(cir): we shouldn't need this but we currently reuse intrinsic IDs for
// convenience.
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/IR/Intrinsics.h"

#include "clang/AST/GlobalDecl.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Frontend/FrontendDiagnostic.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;
using namespace llvm;

static RValue buildLibraryCall(CIRGenFunction &cgf, const FunctionDecl *fd,
                               const CallExpr *e,
                               mlir::Operation *calleeValue) {
  auto callee = CIRGenCallee::forDirect(calleeValue, GlobalDecl(fd));
  return cgf.buildCall(e->getCallee()->getType(), callee, e, ReturnValueSlot());
}

template <class Operation>
static RValue buildUnaryFPBuiltin(CIRGenFunction &cgf, const CallExpr &e) {
  auto arg = cgf.buildScalarExpr(e.getArg(0));

  CIRGenFunction::CIRGenFPOptionsRAII fpOptsRaii(cgf, &e);
  if (cgf.getBuilder().getIsFPConstrained())
    llvm_unreachable("constraint FP operations are NYI");

  auto call =
      cgf.getBuilder().create<Operation>(arg.getLoc(), arg.getType(), arg);
  return RValue::get(call->getResult(0));
}

template <typename Op>
static RValue buildUnaryMaybeConstrainedFPToIntBuiltin(CIRGenFunction &cgf,
                                                       const CallExpr &e) {
  auto resultType = cgf.ConvertType(e.getType());
  auto src = cgf.buildScalarExpr(e.getArg(0));

  if (cgf.getBuilder().getIsFPConstrained())
    llvm_unreachable("constraint FP operations are NYI");

  auto call = cgf.getBuilder().create<Op>(src.getLoc(), resultType, src);
  return RValue::get(call->getResult(0));
}

template <typename Op>
static RValue buildBinaryFPBuiltin(CIRGenFunction &cgf, const CallExpr &e) {
  auto arg0 = cgf.buildScalarExpr(e.getArg(0));
  auto arg1 = cgf.buildScalarExpr(e.getArg(1));

  auto loc = cgf.getLoc(e.getExprLoc());
  auto ty = cgf.ConvertType(e.getType());
  auto call = cgf.getBuilder().create<Op>(loc, ty, arg0, arg1);

  return RValue::get(call->getResult(0));
}

template <typename Op>
static mlir::Value buildBinaryMaybeConstrainedFPBuiltin(CIRGenFunction &cgf,
                                                        const CallExpr &e) {
  auto arg0 = cgf.buildScalarExpr(e.getArg(0));
  auto arg1 = cgf.buildScalarExpr(e.getArg(1));

  auto loc = cgf.getLoc(e.getExprLoc());
  auto ty = cgf.ConvertType(e.getType());

  if (cgf.getBuilder().getIsFPConstrained()) {
    CIRGenFunction::CIRGenFPOptionsRAII fpOptsRaii(cgf, &e);
    llvm_unreachable("constrained FP operations are NYI");
  } else {
    auto call = cgf.getBuilder().create<Op>(loc, ty, arg0, arg1);
    return call->getResult(0);
  }
}

template <typename Op>
static RValue
buildBuiltinBitOp(CIRGenFunction &cgf, const CallExpr *e,
                  std::optional<CIRGenFunction::BuiltinCheckKind> ck) {
  mlir::Value arg;
  if (ck.has_value())
    arg = cgf.buildCheckedArgForBuiltin(e->getArg(0), *ck);
  else
    arg = cgf.buildScalarExpr(e->getArg(0));

  auto resultTy = cgf.ConvertType(e->getType());
  auto op =
      cgf.getBuilder().create<Op>(cgf.getLoc(e->getExprLoc()), resultTy, arg);
  return RValue::get(op);
}

// Initialize the alloca with the given size and alignment according to the lang
// opts. Supporting only the trivial non-initialization for now.
static void initializeAlloca(CIRGenFunction &cgf,
                             [[maybe_unused]] mlir::Value allocaAddr,
                             [[maybe_unused]] mlir::Value size,
                             [[maybe_unused]] CharUnits alignmentInBytes) {

  switch (cgf.getLangOpts().getTrivialAutoVarInit()) {
  case LangOptions::TrivialAutoVarInitKind::Uninitialized:
    // Nothing to initialize.
    return;
  case LangOptions::TrivialAutoVarInitKind::Zero:
  case LangOptions::TrivialAutoVarInitKind::Pattern:
    assert(false && "unexpected trivial auto var init kind NYI");
    return;
  }
}

namespace {
struct WidthAndSignedness {
  unsigned width;
  bool Signed;
};
} // namespace

static WidthAndSignedness
getIntegerWidthAndSignedness(const clang::ASTContext &context,
                             const clang::QualType type) {
  assert(type->isIntegerType() && "Given type is not an integer.");
  unsigned width = type->isBooleanType()  ? 1
                   : type->isBitIntType() ? context.getIntWidth(type)
                                          : context.getTypeInfo(type).Width;
  bool Signed = type->isSignedIntegerType();
  return {width, Signed};
}

// Given one or more integer types, this function produces an integer type that
// encompasses them: any value in one of the given types could be expressed in
// the encompassing type.
static struct WidthAndSignedness
encompassingIntegerType(ArrayRef<struct WidthAndSignedness> Types) {
  assert(!Types.empty() && "Empty list of types.");

  // If any of the given types is signed, we must return a signed type.
  bool Signed = false;
  for (const auto &type : Types) {
    Signed |= type.Signed;
  }

  // The encompassing type must have a width greater than or equal to the width
  // of the specified types.  Additionally, if the encompassing type is signed,
  // its width must be strictly greater than the width of any unsigned types
  // given.
  unsigned width = 0;
  for (const auto &type : Types) {
    unsigned minWidth = type.width + (Signed && !type.Signed);
    if (width < minWidth) {
      width = minWidth;
    }
  }

  return {width, Signed};
}

/// Emit the conversions required to turn the given value into an
/// integer of the given size.
static mlir::Value buildToInt(CIRGenFunction &cgf, mlir::Value v, QualType t,
                              mlir::cir::IntType intType) {
  v = cgf.buildToMemory(v, t);

  if (isa<mlir::cir::PointerType>(v.getType()))
    return cgf.getBuilder().createPtrToInt(v, intType);

  assert(v.getType() == intType);
  return v;
}

static mlir::Value buildFromInt(CIRGenFunction &cgf, mlir::Value v, QualType t,
                                mlir::Type resultType) {
  v = cgf.buildFromMemory(v, t);

  if (isa<mlir::cir::PointerType>(resultType))
    return cgf.getBuilder().createIntToPtr(v, resultType);

  assert(v.getType() == resultType);
  return v;
}

static Address checkAtomicAlignment(CIRGenFunction &cgf, const CallExpr *e) {
  ASTContext &ctx = cgf.getContext();
  Address ptr = cgf.buildPointerWithAlignment(e->getArg(0));
  unsigned bytes =
      isa<mlir::cir::PointerType>(ptr.getElementType())
          ? ctx.getTypeSizeInChars(ctx.VoidPtrTy).getQuantity()
          : cgf.cgm.getDataLayout().getTypeSizeInBits(ptr.getElementType()) / 8;
  unsigned align = ptr.getAlignment().getQuantity();
  if (align % bytes != 0) {
    DiagnosticsEngine &diags = cgf.cgm.getDiags();
    diags.Report(e->getBeginLoc(), diag::warn_sync_op_misaligned);
    // Force address to be at least naturally-aligned.
    return ptr.withAlignment(CharUnits::fromQuantity(bytes));
  }
  return ptr;
}

/// Utility to insert an atomic instruction based on Intrinsic::ID
/// and the expression node.
static mlir::Value
makeBinaryAtomicValue(CIRGenFunction &cgf, mlir::cir::AtomicFetchKind kind,
                      const CallExpr *expr,
                      mlir::cir::MemOrder ordering =
                          mlir::cir::MemOrder::SequentiallyConsistent) {

  QualType typ = expr->getType();

  assert(expr->getArg(0)->getType()->isPointerType());
  assert(cgf.getContext().hasSameUnqualifiedType(
      typ, expr->getArg(0)->getType()->getPointeeType()));
  assert(
      cgf.getContext().hasSameUnqualifiedType(typ, expr->getArg(1)->getType()));

  Address destAddr = checkAtomicAlignment(cgf, expr);
  auto &builder = cgf.getBuilder();
  auto intType =
      expr->getArg(0)->getType()->getPointeeType()->isUnsignedIntegerType()
          ? builder.getUIntNTy(cgf.getContext().getTypeSize(typ))
          : builder.getSIntNTy(cgf.getContext().getTypeSize(typ));
  mlir::Value val = cgf.buildScalarExpr(expr->getArg(1));
  mlir::Type valueType = val.getType();
  val = buildToInt(cgf, val, typ, intType);

  auto rmwi = builder.create<mlir::cir::AtomicFetch>(
      cgf.getLoc(expr->getSourceRange()), destAddr.emitRawPointer(), val, kind,
      ordering, false, /* is volatile */
      true);           /* fetch first */
  return buildFromInt(cgf, rmwi->getResult(0), typ, valueType);
}

static RValue buildBinaryAtomic(CIRGenFunction &cgf,
                                mlir::cir::AtomicFetchKind kind,
                                const CallExpr *e) {
  return RValue::get(makeBinaryAtomicValue(cgf, kind, e));
}

static mlir::Value makeAtomicCmpXchgValue(CIRGenFunction &cgf,
                                          const CallExpr *expr,
                                          bool returnBool) {
  QualType typ = returnBool ? expr->getArg(1)->getType() : expr->getType();
  Address destAddr = checkAtomicAlignment(cgf, expr);
  auto &builder = cgf.getBuilder();

  auto intType = builder.getSIntNTy(cgf.getContext().getTypeSize(typ));
  auto cmpVal = cgf.buildScalarExpr(expr->getArg(1));
  cmpVal = buildToInt(cgf, cmpVal, typ, intType);
  auto newVal =
      buildToInt(cgf, cgf.buildScalarExpr(expr->getArg(2)), typ, intType);

  auto op = builder.create<mlir::cir::AtomicCmpXchg>(
      cgf.getLoc(expr->getSourceRange()), cmpVal.getType(), builder.getBoolTy(),
      destAddr.getPointer(), cmpVal, newVal,
      mlir::cir::MemOrder::SequentiallyConsistent,
      mlir::cir::MemOrder::SequentiallyConsistent);

  return returnBool ? op.getResult(1) : op.getResult(0);
}

RValue CIRGenFunction::buildRotate(const CallExpr *e, bool isRotateRight) {
  auto src = buildScalarExpr(e->getArg(0));
  auto shiftAmt = buildScalarExpr(e->getArg(1));

  // The builtin's shift arg may have a different type than the source arg and
  // result, but the CIR ops uses the same type for all values.
  auto ty = src.getType();
  shiftAmt = builder.createIntCast(shiftAmt, ty);
  auto r = builder.create<mlir::cir::RotateOp>(getLoc(e->getSourceRange()), src,
                                               shiftAmt);
  if (!isRotateRight)
    r->setAttr("left", mlir::UnitAttr::get(src.getContext()));
  return RValue::get(r);
}

RValue CIRGenFunction::buildBuiltinExpr(const GlobalDecl gd, unsigned builtinId,
                                        const CallExpr *e,
                                        ReturnValueSlot returnValue) {
  const FunctionDecl *fd = gd.getDecl()->getAsFunction();

  // See if we can constant fold this builtin.  If so, don't emit it at all.
  // TODO: Extend this handling to all builtin calls that we can constant-fold.
  Expr::EvalResult result;
  if (e->isPRValue() && e->EvaluateAsRValue(result, cgm.getASTContext()) &&
      !result.hasSideEffects()) {
    if (result.Val.isInt()) {
      return RValue::get(builder.getConstInt(getLoc(e->getSourceRange()),
                                             result.Val.getInt()));
    }
    if (result.Val.isFloat()) {
      // Note: we are using result type of CallExpr to determine the type of
      // the constant. Clang Codegen uses the result value to make judgement
      // of the type. We feel it should be Ok to use expression type because
      // it is hard to imagine a builtin function evaluates to
      // a value that over/underflows its own defined type.
      mlir::Type resTy = getCIRType(e->getType());
      return RValue::get(builder.getConstFP(getLoc(e->getExprLoc()), resTy,
                                            result.Val.getFloat()));
    }
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
  const unsigned builtinIdIfNoAsmLabel =
      fd->hasAttr<AsmLabelAttr>() ? 0 : builtinId;

  std::optional<bool> errnoOverriden;
  // ErrnoOverriden is true if math-errno is overriden via the
  // '#pragma float_control(precise, on)'. This pragma disables fast-math,
  // which implies math-errno.
  if (e->hasStoredFPFeatures()) {
    FPOptionsOverride op = e->getFPFeatures();
    if (op.hasMathErrnoOverride())
      errnoOverriden = op.getMathErrnoOverride();
  }
  // True if 'atttibute__((optnone)) is used. This attibute overrides
  // fast-math which implies math-errno.
  bool optNone = CurFuncDecl && CurFuncDecl->hasAttr<OptimizeNoneAttr>();

  // True if we are compiling at -O2 and errno has been disabled
  // using the '#pragma float_control(precise, off)', and
  // attribute opt-none hasn't been seen.
  [[maybe_unused]] bool errnoOverridenToFalseWithOpt =
      errnoOverriden.has_value() && !errnoOverriden.value() && !optNone &&
      cgm.getCodeGenOpts().OptimizationLevel != 0;

  // There are LLVM math intrinsics/instructions corresponding to math library
  // functions except the LLVM op will never set errno while the math library
  // might. Also, math builtins have the same semantics as their math library
  // twins. Thus, we can transform math library and builtin calls to their
  // LLVM counterparts if the call is marked 'const' (known to never set errno).
  // In case FP exceptions are enabled, the experimental versions of the
  // intrinsics model those.
  [[maybe_unused]] bool constAlways =
      getContext().BuiltinInfo.isConst(builtinId);

  // There's a special case with the fma builtins where they are always const
  // if the target environment is GNU or the target is OS is Windows and we're
  // targeting the MSVCRT.dll environment.
  // FIXME: This list can be become outdated. Need to find a way to get it some
  // other way.
  switch (builtinId) {
  case Builtin::BI__builtin_fma:
  case Builtin::BI__builtin_fmaf:
  case Builtin::BI__builtin_fmal:
  case Builtin::BIfma:
  case Builtin::BIfmaf:
  case Builtin::BIfmal: {
    auto &trip = cgm.getTriple();
    if (trip.isGNUEnvironment() || trip.isOSMSVCRT())
      constAlways = true;
    break;
  }
  default:
    break;
  }

  bool constWithoutErrnoAndExceptions =
      getContext().BuiltinInfo.isConstWithoutErrnoAndExceptions(builtinId);
  bool constWithoutExceptions =
      getContext().BuiltinInfo.isConstWithoutExceptions(builtinId);

  // ConstAttr is enabled in fast-math mode. In fast-math mode, math-errno is
  // disabled.
  // Math intrinsics are generated only when math-errno is disabled. Any pragmas
  // or attributes that affect math-errno should prevent or allow math
  // intrincs to be generated. Intrinsics are generated:
  //   1- In fast math mode, unless math-errno is overriden
  //      via '#pragma float_control(precise, on)', or via an
  //      'attribute__((optnone))'.
  //   2- If math-errno was enabled on command line but overriden
  //      to false via '#pragma float_control(precise, off))' and
  //      'attribute__((optnone))' hasn't been used.
  //   3- If we are compiling with optimization and errno has been disabled
  //      via '#pragma float_control(precise, off)', and
  //      'attribute__((optnone))' hasn't been used.

  bool constWithoutErrnoOrExceptions =
      constWithoutErrnoAndExceptions || constWithoutExceptions;
  bool generateIntrinsics =
      (constAlways && !optNone) ||
      (!getLangOpts().MathErrno &&
       !(errnoOverriden.has_value() && errnoOverriden.value()) && !optNone);
  if (!generateIntrinsics) {
    generateIntrinsics =
        constWithoutErrnoOrExceptions && !constWithoutErrnoAndExceptions;
    if (!generateIntrinsics)
      generateIntrinsics =
          constWithoutErrnoOrExceptions &&
          (!getLangOpts().MathErrno &&
           !(errnoOverriden.has_value() && errnoOverriden.value()) && !optNone);
    if (!generateIntrinsics)
      generateIntrinsics =
          constWithoutErrnoOrExceptions && errnoOverridenToFalseWithOpt;
  }

  if (generateIntrinsics) {
    switch (builtinIdIfNoAsmLabel) {
    case Builtin::BIceil:
    case Builtin::BIceilf:
    case Builtin::BIceill:
    case Builtin::BI__builtin_ceil:
    case Builtin::BI__builtin_ceilf:
    case Builtin::BI__builtin_ceilf16:
    case Builtin::BI__builtin_ceill:
    case Builtin::BI__builtin_ceilf128:
      return buildUnaryFPBuiltin<mlir::cir::CeilOp>(*this, *e);

    case Builtin::BIcopysign:
    case Builtin::BIcopysignf:
    case Builtin::BIcopysignl:
    case Builtin::BI__builtin_copysign:
    case Builtin::BI__builtin_copysignf:
    case Builtin::BI__builtin_copysignl:
      return buildBinaryFPBuiltin<mlir::cir::CopysignOp>(*this, *e);

    case Builtin::BI__builtin_copysignf16:
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
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::CosOp>(*this, *e);

    case Builtin::BIexp:
    case Builtin::BIexpf:
    case Builtin::BIexpl:
    case Builtin::BI__builtin_exp:
    case Builtin::BI__builtin_expf:
    case Builtin::BI__builtin_expf16:
    case Builtin::BI__builtin_expl:
    case Builtin::BI__builtin_expf128:
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::ExpOp>(*this, *e);

    case Builtin::BIexp2:
    case Builtin::BIexp2f:
    case Builtin::BIexp2l:
    case Builtin::BI__builtin_exp2:
    case Builtin::BI__builtin_exp2f:
    case Builtin::BI__builtin_exp2f16:
    case Builtin::BI__builtin_exp2l:
    case Builtin::BI__builtin_exp2f128:
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::Exp2Op>(*this, *e);

    case Builtin::BIfabs:
    case Builtin::BIfabsf:
    case Builtin::BIfabsl:
    case Builtin::BI__builtin_fabs:
    case Builtin::BI__builtin_fabsf:
    case Builtin::BI__builtin_fabsf16:
    case Builtin::BI__builtin_fabsl:
    case Builtin::BI__builtin_fabsf128:
      return buildUnaryFPBuiltin<mlir::cir::FAbsOp>(*this, *e);

    case Builtin::BIfloor:
    case Builtin::BIfloorf:
    case Builtin::BIfloorl:
    case Builtin::BI__builtin_floor:
    case Builtin::BI__builtin_floorf:
    case Builtin::BI__builtin_floorf16:
    case Builtin::BI__builtin_floorl:
    case Builtin::BI__builtin_floorf128:
      return buildUnaryFPBuiltin<mlir::cir::FloorOp>(*this, *e);

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
    case Builtin::BI__builtin_fmaxl:
      return RValue::get(
          buildBinaryMaybeConstrainedFPBuiltin<mlir::cir::FMaxOp>(*this, *e));

    case Builtin::BI__builtin_fmaxf16:
    case Builtin::BI__builtin_fmaxf128:
      llvm_unreachable("NYI");

    case Builtin::BIfmin:
    case Builtin::BIfminf:
    case Builtin::BIfminl:
    case Builtin::BI__builtin_fmin:
    case Builtin::BI__builtin_fminf:
    case Builtin::BI__builtin_fminl:
      return RValue::get(
          buildBinaryMaybeConstrainedFPBuiltin<mlir::cir::FMinOp>(*this, *e));

    case Builtin::BI__builtin_fminf16:
    case Builtin::BI__builtin_fminf128:
      llvm_unreachable("NYI");

    // fmod() is a special-case. It maps to the frem instruction rather than an
    // LLVM intrinsic.
    case Builtin::BIfmod:
    case Builtin::BIfmodf:
    case Builtin::BIfmodl:
    case Builtin::BI__builtin_fmod:
    case Builtin::BI__builtin_fmodf:
    case Builtin::BI__builtin_fmodl:
      assert(!MissingFeatures::fastMathFlags());
      return buildBinaryFPBuiltin<mlir::cir::FModOp>(*this, *e);

    case Builtin::BI__builtin_fmodf16:
    case Builtin::BI__builtin_fmodf128:
      llvm_unreachable("NYI");

    case Builtin::BIlog:
    case Builtin::BIlogf:
    case Builtin::BIlogl:
    case Builtin::BI__builtin_log:
    case Builtin::BI__builtin_logf:
    case Builtin::BI__builtin_logf16:
    case Builtin::BI__builtin_logl:
    case Builtin::BI__builtin_logf128:
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::LogOp>(*this, *e);

    case Builtin::BIlog10:
    case Builtin::BIlog10f:
    case Builtin::BIlog10l:
    case Builtin::BI__builtin_log10:
    case Builtin::BI__builtin_log10f:
    case Builtin::BI__builtin_log10f16:
    case Builtin::BI__builtin_log10l:
    case Builtin::BI__builtin_log10f128:
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::Log10Op>(*this, *e);

    case Builtin::BIlog2:
    case Builtin::BIlog2f:
    case Builtin::BIlog2l:
    case Builtin::BI__builtin_log2:
    case Builtin::BI__builtin_log2f:
    case Builtin::BI__builtin_log2f16:
    case Builtin::BI__builtin_log2l:
    case Builtin::BI__builtin_log2f128:
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::Log2Op>(*this, *e);

    case Builtin::BInearbyint:
    case Builtin::BInearbyintf:
    case Builtin::BInearbyintl:
    case Builtin::BI__builtin_nearbyint:
    case Builtin::BI__builtin_nearbyintf:
    case Builtin::BI__builtin_nearbyintl:
    case Builtin::BI__builtin_nearbyintf128:
      return buildUnaryFPBuiltin<mlir::cir::NearbyintOp>(*this, *e);

    case Builtin::BIpow:
    case Builtin::BIpowf:
    case Builtin::BIpowl:
    case Builtin::BI__builtin_pow:
    case Builtin::BI__builtin_powf:
    case Builtin::BI__builtin_powl:
      assert(!MissingFeatures::fastMathFlags());
      return RValue::get(
          buildBinaryMaybeConstrainedFPBuiltin<mlir::cir::PowOp>(*this, *e));

    case Builtin::BI__builtin_powf16:
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
      return buildUnaryFPBuiltin<mlir::cir::RintOp>(*this, *e);

    case Builtin::BIround:
    case Builtin::BIroundf:
    case Builtin::BIroundl:
    case Builtin::BI__builtin_round:
    case Builtin::BI__builtin_roundf:
    case Builtin::BI__builtin_roundf16:
    case Builtin::BI__builtin_roundl:
    case Builtin::BI__builtin_roundf128:
      return buildUnaryFPBuiltin<mlir::cir::RoundOp>(*this, *e);

    case Builtin::BIsin:
    case Builtin::BIsinf:
    case Builtin::BIsinl:
    case Builtin::BI__builtin_sin:
    case Builtin::BI__builtin_sinf:
    case Builtin::BI__builtin_sinf16:
    case Builtin::BI__builtin_sinl:
    case Builtin::BI__builtin_sinf128:
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::SinOp>(*this, *e);

    case Builtin::BIsqrt:
    case Builtin::BIsqrtf:
    case Builtin::BIsqrtl:
    case Builtin::BI__builtin_sqrt:
    case Builtin::BI__builtin_sqrtf:
    case Builtin::BI__builtin_sqrtf16:
    case Builtin::BI__builtin_sqrtl:
    case Builtin::BI__builtin_sqrtf128:
      assert(!MissingFeatures::fastMathFlags());
      return buildUnaryFPBuiltin<mlir::cir::SqrtOp>(*this, *e);

    case Builtin::BItrunc:
    case Builtin::BItruncf:
    case Builtin::BItruncl:
    case Builtin::BI__builtin_trunc:
    case Builtin::BI__builtin_truncf:
    case Builtin::BI__builtin_truncf16:
    case Builtin::BI__builtin_truncl:
    case Builtin::BI__builtin_truncf128:
      return buildUnaryFPBuiltin<mlir::cir::TruncOp>(*this, *e);

    case Builtin::BIlround:
    case Builtin::BIlroundf:
    case Builtin::BIlroundl:
    case Builtin::BI__builtin_lround:
    case Builtin::BI__builtin_lroundf:
    case Builtin::BI__builtin_lroundl:
      return buildUnaryMaybeConstrainedFPToIntBuiltin<mlir::cir::LroundOp>(
          *this, *e);

    case Builtin::BI__builtin_lroundf128:
      llvm_unreachable("NYI");

    case Builtin::BIllround:
    case Builtin::BIllroundf:
    case Builtin::BIllroundl:
    case Builtin::BI__builtin_llround:
    case Builtin::BI__builtin_llroundf:
    case Builtin::BI__builtin_llroundl:
      return buildUnaryMaybeConstrainedFPToIntBuiltin<mlir::cir::LLroundOp>(
          *this, *e);

    case Builtin::BI__builtin_llroundf128:
      llvm_unreachable("NYI");

    case Builtin::BIlrint:
    case Builtin::BIlrintf:
    case Builtin::BIlrintl:
    case Builtin::BI__builtin_lrint:
    case Builtin::BI__builtin_lrintf:
    case Builtin::BI__builtin_lrintl:
      return buildUnaryMaybeConstrainedFPToIntBuiltin<mlir::cir::LrintOp>(*this,
                                                                          *e);

    case Builtin::BI__builtin_lrintf128:
      llvm_unreachable("NYI");

    case Builtin::BIllrint:
    case Builtin::BIllrintf:
    case Builtin::BIllrintl:
    case Builtin::BI__builtin_llrint:
    case Builtin::BI__builtin_llrintf:
    case Builtin::BI__builtin_llrintl:
      return buildUnaryMaybeConstrainedFPToIntBuiltin<mlir::cir::LLrintOp>(
          *this, *e);

    case Builtin::BI__builtin_llrintf128:
      llvm_unreachable("NYI");

    default:
      break;
    }
  }

  switch (builtinIdIfNoAsmLabel) {
  default:
    break;

  case Builtin::BI__builtin_complex: {
    mlir::Value real = buildScalarExpr(e->getArg(0));
    mlir::Value imag = buildScalarExpr(e->getArg(1));
    mlir::Value complex =
        builder.createComplexCreate(getLoc(e->getExprLoc()), real, imag);
    return RValue::getComplex(complex);
  }

  case Builtin::BI__builtin_creal:
  case Builtin::BI__builtin_crealf:
  case Builtin::BI__builtin_creall:
  case Builtin::BIcreal:
  case Builtin::BIcrealf:
  case Builtin::BIcreall: {
    mlir::Value complexVal = buildComplexExpr(e->getArg(0));
    mlir::Value real =
        builder.createComplexReal(getLoc(e->getExprLoc()), complexVal);
    return RValue::get(real);
  }

  case Builtin::BI__builtin_cimag:
  case Builtin::BI__builtin_cimagf:
  case Builtin::BI__builtin_cimagl:
  case Builtin::BIcimag:
  case Builtin::BIcimagf:
  case Builtin::BIcimagl: {
    mlir::Value complexVal = buildComplexExpr(e->getArg(0));
    mlir::Value real =
        builder.createComplexImag(getLoc(e->getExprLoc()), complexVal);
    return RValue::get(real);
  }

  case Builtin::BI__builtin_conj:
  case Builtin::BI__builtin_conjf:
  case Builtin::BI__builtin_conjl:
  case Builtin::BIconj:
  case Builtin::BIconjf:
  case Builtin::BIconjl: {
    mlir::Value complexVal = buildComplexExpr(e->getArg(0));
    mlir::Value conj = builder.createUnaryOp(
        getLoc(e->getExprLoc()), mlir::cir::UnaryOpKind::Not, complexVal);
    return RValue::getComplex(conj);
  }

  case Builtin::BI__builtin___CFStringMakeConstantString:
  case Builtin::BI__builtin___NSStringMakeConstantString:
    llvm_unreachable("NYI");

  case Builtin::BIprintf:
    if (getTarget().getTriple().isNVPTX() ||
        getTarget().getTriple().isAMDGCN()) {
      llvm_unreachable("NYI");
    }
    break;

  // C stdarg builtins.
  case Builtin::BI__builtin_stdarg_start:
  case Builtin::BI__builtin_va_start:
  case Builtin::BI__va_start:
  case Builtin::BI__builtin_va_end: {
    buildVAStartEnd(builtinId == Builtin::BI__va_start
                        ? buildScalarExpr(e->getArg(0))
                        : buildVAListRef(e->getArg(0)).getPointer(),
                    builtinId != Builtin::BI__builtin_va_end);
    return {};
  }
  case Builtin::BI__builtin_va_copy: {
    auto dstPtr = buildVAListRef(e->getArg(0)).getPointer();
    auto srcPtr = buildVAListRef(e->getArg(1)).getPointer();
    builder.create<mlir::cir::VACopyOp>(dstPtr.getLoc(), dstPtr, srcPtr);
    return {};
  }

  case Builtin::BI__builtin_expect:
  case Builtin::BI__builtin_expect_with_probability: {
    auto argValue = buildScalarExpr(e->getArg(0));
    auto expectedValue = buildScalarExpr(e->getArg(1));

    // Don't generate cir.expect on -O0 as the backend won't use it for
    // anything. Note, we still IRGen ExpectedValue because it could have
    // side-effects.
    if (cgm.getCodeGenOpts().OptimizationLevel == 0)
      return RValue::get(argValue);

    mlir::FloatAttr probAttr = {};
    if (builtinIdIfNoAsmLabel == Builtin::BI__builtin_expect_with_probability) {
      llvm::APFloat probability(0.0);
      const Expr *probArg = e->getArg(2);
      bool evalSucceed =
          probArg->EvaluateAsFloat(probability, cgm.getASTContext());
      assert(evalSucceed && "probability should be able to evaluate as float");
      (void)evalSucceed;
      bool loseInfo = false;
      probability.convert(llvm::APFloat::IEEEdouble(),
                          llvm::RoundingMode::Dynamic, &loseInfo);
      probAttr = mlir::FloatAttr::get(
          mlir::FloatType::getF64(builder.getContext()), probability);
    }

    auto result = builder.create<mlir::cir::ExpectOp>(
        getLoc(e->getSourceRange()), argValue.getType(), argValue,
        expectedValue, probAttr);

    return RValue::get(result);
  }
  case Builtin::BI__builtin_unpredictable: {
    if (cgm.getCodeGenOpts().OptimizationLevel != 0)
      assert(!MissingFeatures::insertBuiltinUnpredictable());
    return RValue::get(buildScalarExpr(e->getArg(0)));
  }

  case Builtin::BI__builtin_assume_aligned: {
    const Expr *ptr = e->getArg(0);
    mlir::Value ptrValue = buildScalarExpr(ptr);
    mlir::Value offsetValue =
        (e->getNumArgs() > 2) ? buildScalarExpr(e->getArg(2)) : nullptr;

    mlir::Attribute alignmentAttr = ConstantEmitter(*this).emitAbstract(
        e->getArg(1), e->getArg(1)->getType());
    std::int64_t alignment = cast<mlir::cir::IntAttr>(alignmentAttr).getSInt();

    ptrValue = buildAlignmentAssumption(ptrValue, ptr, ptr->getExprLoc(),
                                        builder.getI64IntegerAttr(alignment),
                                        offsetValue);
    return RValue::get(ptrValue);
  }

  case Builtin::BI__assume:
  case Builtin::BI__builtin_assume: {
    if (e->getArg(0)->HasSideEffects(getContext()))
      return RValue::get(nullptr);

    mlir::Value argValue = buildScalarExpr(e->getArg(0));
    builder.create<mlir::cir::AssumeOp>(getLoc(e->getExprLoc()), argValue);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_assume_separate_storage: {
    const Expr *arg0 = e->getArg(0);
    const Expr *arg1 = e->getArg(1);

    mlir::Value value0 = buildScalarExpr(arg0);
    mlir::Value value1 = buildScalarExpr(arg1);

    builder.create<mlir::cir::AssumeSepStorageOp>(getLoc(e->getExprLoc()),
                                                  value0, value1);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin_prefetch: {
    auto evaluateOperandAsInt = [&](const Expr *arg) {
      Expr::EvalResult res;
      [[maybe_unused]] bool evalSucceed =
          arg->EvaluateAsInt(res, cgm.getASTContext());
      assert(evalSucceed && "expression should be able to evaluate as int");
      return res.Val.getInt().getZExtValue();
    };

    bool isWrite = false;
    if (e->getNumArgs() > 1)
      isWrite = evaluateOperandAsInt(e->getArg(1));

    int locality = 0;
    if (e->getNumArgs() > 2)
      locality = evaluateOperandAsInt(e->getArg(2));

    mlir::Value address = buildScalarExpr(e->getArg(0));
    builder.create<mlir::cir::PrefetchOp>(getLoc(e->getSourceRange()), address,
                                          locality, isWrite);
    return RValue::get(nullptr);
  }

  case Builtin::BI__builtin___clear_cache: {
    mlir::Type voidTy = mlir::cir::VoidType::get(builder.getContext());
    mlir::Value begin =
        builder.createPtrBitcast(buildScalarExpr(e->getArg(0)), voidTy);
    mlir::Value end =
        builder.createPtrBitcast(buildScalarExpr(e->getArg(1)), voidTy);
    builder.create<mlir::cir::ClearCacheOp>(getLoc(e->getSourceRange()), begin,
                                            end);
    return RValue::get(nullptr);
  }

  // C++ std:: builtins.
  case Builtin::BImove:
  case Builtin::BImove_if_noexcept:
  case Builtin::BIforward:
  case Builtin::BIas_const:
    return RValue::get(buildLValue(e->getArg(0)).getPointer());
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
    GlobalDecl gd{fd};
    mlir::Type ty = cgm.getTypes().GetFunctionType(
        cgm.getTypes().arrangeGlobalDeclaration(gd));
    const auto *nd = cast<NamedDecl>(gd.getDecl());
    auto fnOp =
        cgm.GetOrCreateCIRFunction(nd->getName(), ty, gd, /*ForVTable=*/false,
                                   /*DontDefer=*/false);
    fnOp.setBuiltinAttr(mlir::UnitAttr::get(builder.getContext()));
    return buildCall(e->getCallee()->getType(), CIRGenCallee::forDirect(fnOp),
                     e, returnValue);
  }
  case Builtin::BI__builtin_dynamic_object_size: {
    // Fallthrough below, assert until we have a testcase.
    llvm_unreachable("NYI");
  }
  case Builtin::BI__builtin_object_size: {
    unsigned type =
        e->getArg(1)->EvaluateKnownConstInt(getContext()).getZExtValue();
    auto resType =
        mlir::dyn_cast<mlir::cir::IntType>(ConvertType(e->getType()));
    assert(resType && "not sure what to do?");

    // We pass this builtin onto the optimizer so that it can figure out the
    // object size in more complex cases.
    bool isDynamic = builtinId == Builtin::BI__builtin_dynamic_object_size;
    return RValue::get(emitBuiltinObjectSize(e->getArg(0), type, resType,
                                             /*EmittedE=*/nullptr, isDynamic));
  }
  case Builtin::BI__builtin_unreachable: {
    buildUnreachable(e->getExprLoc());

    // We do need to preserve an insertion point.
    builder.createBlock(builder.getBlock()->getParent());

    return RValue::get(nullptr);
  }
  case Builtin::BI__builtin_trap: {
    builder.create<mlir::cir::TrapOp>(getLoc(e->getExprLoc()));

    // Note that cir.trap is a terminator so we need to start a new block to
    // preserve the insertion point.
    builder.createBlock(builder.getBlock()->getParent());

    return RValue::get(nullptr);
  }
  case Builtin::BImemcpy:
  case Builtin::BI__builtin_memcpy:
  case Builtin::BImempcpy:
  case Builtin::BI__builtin_mempcpy: {
    Address dest = buildPointerWithAlignment(e->getArg(0));
    Address src = buildPointerWithAlignment(e->getArg(1));
    mlir::Value sizeVal = buildScalarExpr(e->getArg(2));
    buildNonNullArgCheck(RValue::get(dest.getPointer()),
                         e->getArg(0)->getType(), e->getArg(0)->getExprLoc(),
                         fd, 0);
    buildNonNullArgCheck(RValue::get(src.getPointer()), e->getArg(1)->getType(),
                         e->getArg(1)->getExprLoc(), fd, 1);
    builder.createMemCpy(getLoc(e->getSourceRange()), dest.getPointer(),
                         src.getPointer(), sizeVal);
    if (builtinId == Builtin::BImempcpy ||
        builtinId == Builtin::BI__builtin_mempcpy)
      llvm_unreachable("mempcpy is NYI");
    else
      return RValue::get(dest.getPointer());
  }

  case Builtin::BI__builtin_clrsb:
  case Builtin::BI__builtin_clrsbl:
  case Builtin::BI__builtin_clrsbll:
    return buildBuiltinBitOp<mlir::cir::BitClrsbOp>(*this, e, std::nullopt);

  case Builtin::BI__builtin_ctzs:
  case Builtin::BI__builtin_ctz:
  case Builtin::BI__builtin_ctzl:
  case Builtin::BI__builtin_ctzll:
  case Builtin::BI__builtin_ctzg:
    return buildBuiltinBitOp<mlir::cir::BitCtzOp>(*this, e, BCK_CTZPassedZero);

  case Builtin::BI__builtin_clzs:
  case Builtin::BI__builtin_clz:
  case Builtin::BI__builtin_clzl:
  case Builtin::BI__builtin_clzll:
  case Builtin::BI__builtin_clzg:
    return buildBuiltinBitOp<mlir::cir::BitClzOp>(*this, e, BCK_CLZPassedZero);

  case Builtin::BI__builtin_ffs:
  case Builtin::BI__builtin_ffsl:
  case Builtin::BI__builtin_ffsll:
    return buildBuiltinBitOp<mlir::cir::BitFfsOp>(*this, e, std::nullopt);

  case Builtin::BI__builtin_parity:
  case Builtin::BI__builtin_parityl:
  case Builtin::BI__builtin_parityll:
    return buildBuiltinBitOp<mlir::cir::BitParityOp>(*this, e, std::nullopt);

  case Builtin::BI__popcnt16:
  case Builtin::BI__popcnt:
  case Builtin::BI__popcnt64:
  case Builtin::BI__builtin_popcount:
  case Builtin::BI__builtin_popcountl:
  case Builtin::BI__builtin_popcountll:
  case Builtin::BI__builtin_popcountg:
    return buildBuiltinBitOp<mlir::cir::BitPopcountOp>(*this, e, std::nullopt);

  case Builtin::BI__builtin_bswap16:
  case Builtin::BI__builtin_bswap32:
  case Builtin::BI__builtin_bswap64:
  case Builtin::BI_byteswap_ushort:
  case Builtin::BI_byteswap_ulong:
  case Builtin::BI_byteswap_uint64: {
    auto arg = buildScalarExpr(e->getArg(0));
    return RValue::get(builder.create<mlir::cir::ByteswapOp>(
        getLoc(e->getSourceRange()), arg));
  }

  case Builtin::BI__builtin_rotateleft8:
  case Builtin::BI__builtin_rotateleft16:
  case Builtin::BI__builtin_rotateleft32:
  case Builtin::BI__builtin_rotateleft64:
  case Builtin::BI_rotl8: // Microsoft variants of rotate left
  case Builtin::BI_rotl16:
  case Builtin::BI_rotl:
  case Builtin::BI_lrotl:
  case Builtin::BI_rotl64:
    return buildRotate(e, false);

  case Builtin::BI__builtin_rotateright8:
  case Builtin::BI__builtin_rotateright16:
  case Builtin::BI__builtin_rotateright32:
  case Builtin::BI__builtin_rotateright64:
  case Builtin::BI_rotr8: // Microsoft variants of rotate right
  case Builtin::BI_rotr16:
  case Builtin::BI_rotr:
  case Builtin::BI_lrotr:
  case Builtin::BI_rotr64:
    return buildRotate(e, true);

  case Builtin::BI__builtin_constant_p: {
    mlir::Type resultType = ConvertType(e->getType());

    const Expr *arg = e->getArg(0);
    QualType argType = arg->getType();
    // FIXME: The allowance for Obj-C pointers and block pointers is historical
    // and likely a mistake.
    if (!argType->isIntegralOrEnumerationType() && !argType->isFloatingType() &&
        !argType->isObjCObjectPointerType() && !argType->isBlockPointerType())
      // Per the GCC documentation, only numeric constants are recognized after
      // inlining.
      return RValue::get(
          builder.getConstInt(getLoc(e->getSourceRange()),
                              mlir::cast<mlir::cir::IntType>(resultType), 0));

    if (arg->HasSideEffects(getContext()))
      // The argument is unevaluated, so be conservative if it might have
      // side-effects.
      return RValue::get(
          builder.getConstInt(getLoc(e->getSourceRange()),
                              mlir::cast<mlir::cir::IntType>(resultType), 0));

    mlir::Value argValue = buildScalarExpr(arg);
    if (argType->isObjCObjectPointerType())
      // Convert Objective-C objects to id because we cannot distinguish between
      // LLVM types for Obj-C classes as they are opaque.
      argType = cgm.getASTContext().getObjCIdType();
    argValue = builder.createBitcast(argValue, ConvertType(argType));

    mlir::Value result = builder.create<mlir::cir::IsConstantOp>(
        getLoc(e->getSourceRange()), argValue);
    if (result.getType() != resultType)
      result = builder.createBoolToInt(result, resultType);
    return RValue::get(result);
  }

  case Builtin::BIalloca:
  case Builtin::BI_alloca:
  case Builtin::BI__builtin_alloca_uninitialized:
  case Builtin::BI__builtin_alloca: {
    // Get alloca size input
    mlir::Value size = buildScalarExpr(e->getArg(0));

    // The alignment of the alloca should correspond to __BIGGEST_ALIGNMENT__.
    const TargetInfo &ti = getContext().getTargetInfo();
    const CharUnits suitableAlignmentInBytes =
        getContext().toCharUnitsFromBits(ti.getSuitableAlign());

    // Emit the alloca op with type `u8 *` to match the semantics of
    // `llvm.alloca`. We later bitcast the type to `void *` to match the
    // semantics of C/C++
    // FIXME(cir): It may make sense to allow AllocaOp of type `u8` to return a
    // pointer of type `void *`. This will require a change to the allocaOp
    // verifier.
    auto allocaAddr = builder.createAlloca(
        getLoc(e->getSourceRange()), builder.getUInt8PtrTy(),
        builder.getUInt8Ty(), "bi_alloca", suitableAlignmentInBytes, size);

    // Initialize the allocated buffer if required.
    if (builtinId != Builtin::BI__builtin_alloca_uninitialized)
      initializeAlloca(*this, allocaAddr, size, suitableAlignmentInBytes);

    // An alloca will always return a pointer to the alloca (stack) address
    // space. This address space need not be the same as the AST / Language
    // default (e.g. in C / C++ auto vars are in the generic address space). At
    // the AST level this is handled within CreateTempAlloca et al., but for the
    // builtin / dynamic alloca we have to handle it here.
    assert(!MissingFeatures::addressSpace());
    auto aas = getCIRAllocaAddressSpace();
    auto eas = builder.getAddrSpaceAttr(
        e->getType()->getPointeeType().getAddressSpace());
    if (eas != aas) {
      assert(false && "Non-default address space for alloca NYI");
    }

    // Bitcast the alloca to the expected type.
    return RValue::get(
        builder.createBitcast(allocaAddr, builder.getVoidPtrTy()));
  }

  case Builtin::BI__sync_fetch_and_add:
    llvm_unreachable("Shouldn't make it through sema");
  case Builtin::BI__sync_fetch_and_add_1:
  case Builtin::BI__sync_fetch_and_add_2:
  case Builtin::BI__sync_fetch_and_add_4:
  case Builtin::BI__sync_fetch_and_add_8:
  case Builtin::BI__sync_fetch_and_add_16: {
    return buildBinaryAtomic(*this, mlir::cir::AtomicFetchKind::Add, e);
  }

  case Builtin::BI__sync_fetch_and_sub:
    llvm_unreachable("Shouldn't make it through sema");
  case Builtin::BI__sync_fetch_and_sub_1:
  case Builtin::BI__sync_fetch_and_sub_2:
  case Builtin::BI__sync_fetch_and_sub_4:
  case Builtin::BI__sync_fetch_and_sub_8:
  case Builtin::BI__sync_fetch_and_sub_16: {
    return buildBinaryAtomic(*this, mlir::cir::AtomicFetchKind::Sub, e);
  }

  case Builtin::BI__sync_val_compare_and_swap_1:
  case Builtin::BI__sync_val_compare_and_swap_2:
  case Builtin::BI__sync_val_compare_and_swap_4:
  case Builtin::BI__sync_val_compare_and_swap_8:
  case Builtin::BI__sync_val_compare_and_swap_16:
    return RValue::get(makeAtomicCmpXchgValue(*this, e, false));

  case Builtin::BI__sync_bool_compare_and_swap_1:
  case Builtin::BI__sync_bool_compare_and_swap_2:
  case Builtin::BI__sync_bool_compare_and_swap_4:
  case Builtin::BI__sync_bool_compare_and_swap_8:
  case Builtin::BI__sync_bool_compare_and_swap_16:
    return RValue::get(makeAtomicCmpXchgValue(*this, e, true));

  case Builtin::BI__builtin_add_overflow:
  case Builtin::BI__builtin_sub_overflow:
  case Builtin::BI__builtin_mul_overflow: {
    const clang::Expr *leftArg = e->getArg(0);
    const clang::Expr *rightArg = e->getArg(1);
    const clang::Expr *resultArg = e->getArg(2);

    clang::QualType resultQTy =
        resultArg->getType()->castAs<clang::PointerType>()->getPointeeType();

    WidthAndSignedness leftInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), leftArg->getType());
    WidthAndSignedness rightInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), rightArg->getType());
    WidthAndSignedness resultInfo =
        getIntegerWidthAndSignedness(cgm.getASTContext(), resultQTy);

    // Note we compute the encompassing type with the consideration to the
    // result type, so later in LLVM lowering we don't get redundant integral
    // extension casts.
    WidthAndSignedness encompassingInfo =
        encompassingIntegerType({leftInfo, rightInfo, resultInfo});

    auto encompassingCirTy = mlir::cir::IntType::get(
        builder.getContext(), encompassingInfo.width, encompassingInfo.Signed);
    auto resultCirTy =
        mlir::cast<mlir::cir::IntType>(cgm.getTypes().ConvertType(resultQTy));

    mlir::Value left = buildScalarExpr(leftArg);
    mlir::Value right = buildScalarExpr(rightArg);
    Address resultPtr = buildPointerWithAlignment(resultArg);

    // Extend each operand to the encompassing type, if necessary.
    if (left.getType() != encompassingCirTy)
      left = builder.createCast(mlir::cir::CastKind::integral, left,
                                encompassingCirTy);
    if (right.getType() != encompassingCirTy)
      right = builder.createCast(mlir::cir::CastKind::integral, right,
                                 encompassingCirTy);

    // Perform the operation on the extended values.
    mlir::cir::BinOpOverflowKind opKind;
    switch (builtinId) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_add_overflow:
      opKind = mlir::cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_sub_overflow:
      opKind = mlir::cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_mul_overflow:
      opKind = mlir::cir::BinOpOverflowKind::Mul;
      break;
    }

    auto loc = getLoc(e->getSourceRange());
    auto arithResult =
        builder.createBinOpOverflowOp(loc, resultCirTy, opKind, left, right);

    // Here is a slight difference from the original clang CodeGen:
    //   - In the original clang CodeGen, the checked arithmetic result is
    //     first computed as a value of the encompassing type, and then it is
    //     truncated to the actual result type with a second overflow checking.
    //   - In CIRGen, the checked arithmetic operation directly produce the
    //     checked arithmetic result in its expected type.
    //
    // So we don't need a truncation and a second overflow checking here.

    // Finally, store the result using the pointer.
    bool isVolatile =
        resultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(loc, buildToMemory(arithResult.result, resultQTy),
                        resultPtr, isVolatile);

    return RValue::get(arithResult.overflow);
  }

  case Builtin::BI__builtin_uadd_overflow:
  case Builtin::BI__builtin_uaddl_overflow:
  case Builtin::BI__builtin_uaddll_overflow:
  case Builtin::BI__builtin_usub_overflow:
  case Builtin::BI__builtin_usubl_overflow:
  case Builtin::BI__builtin_usubll_overflow:
  case Builtin::BI__builtin_umul_overflow:
  case Builtin::BI__builtin_umull_overflow:
  case Builtin::BI__builtin_umulll_overflow:
  case Builtin::BI__builtin_sadd_overflow:
  case Builtin::BI__builtin_saddl_overflow:
  case Builtin::BI__builtin_saddll_overflow:
  case Builtin::BI__builtin_ssub_overflow:
  case Builtin::BI__builtin_ssubl_overflow:
  case Builtin::BI__builtin_ssubll_overflow:
  case Builtin::BI__builtin_smul_overflow:
  case Builtin::BI__builtin_smull_overflow:
  case Builtin::BI__builtin_smulll_overflow: {
    // Scalarize our inputs.
    mlir::Value x = buildScalarExpr(e->getArg(0));
    mlir::Value y = buildScalarExpr(e->getArg(1));

    const clang::Expr *resultArg = e->getArg(2);
    Address resultPtr = buildPointerWithAlignment(resultArg);

    // Decide which of the arithmetic operation we are lowering to:
    mlir::cir::BinOpOverflowKind arithKind;
    switch (builtinId) {
    default:
      llvm_unreachable("Unknown overflow builtin id.");
    case Builtin::BI__builtin_uadd_overflow:
    case Builtin::BI__builtin_uaddl_overflow:
    case Builtin::BI__builtin_uaddll_overflow:
    case Builtin::BI__builtin_sadd_overflow:
    case Builtin::BI__builtin_saddl_overflow:
    case Builtin::BI__builtin_saddll_overflow:
      arithKind = mlir::cir::BinOpOverflowKind::Add;
      break;
    case Builtin::BI__builtin_usub_overflow:
    case Builtin::BI__builtin_usubl_overflow:
    case Builtin::BI__builtin_usubll_overflow:
    case Builtin::BI__builtin_ssub_overflow:
    case Builtin::BI__builtin_ssubl_overflow:
    case Builtin::BI__builtin_ssubll_overflow:
      arithKind = mlir::cir::BinOpOverflowKind::Sub;
      break;
    case Builtin::BI__builtin_umul_overflow:
    case Builtin::BI__builtin_umull_overflow:
    case Builtin::BI__builtin_umulll_overflow:
    case Builtin::BI__builtin_smul_overflow:
    case Builtin::BI__builtin_smull_overflow:
    case Builtin::BI__builtin_smulll_overflow:
      arithKind = mlir::cir::BinOpOverflowKind::Mul;
      break;
    }

    clang::QualType resultQTy =
        resultArg->getType()->castAs<clang::PointerType>()->getPointeeType();
    auto resultCirTy =
        mlir::cast<mlir::cir::IntType>(cgm.getTypes().ConvertType(resultQTy));

    auto loc = getLoc(e->getSourceRange());
    auto arithResult =
        builder.createBinOpOverflowOp(loc, resultCirTy, arithKind, x, y);

    bool isVolatile =
        resultArg->getType()->getPointeeType().isVolatileQualified();
    builder.createStore(loc, buildToMemory(arithResult.result, resultQTy),
                        resultPtr, isVolatile);

    return RValue::get(arithResult.overflow);
  }
  }

  // If this is an alias for a lib function (e.g. __builtin_sin), emit
  // the call using the normal call path, but using the unmangled
  // version of the function name.
  if (getContext().BuiltinInfo.isLibFunction(builtinId))
    return buildLibraryCall(*this, fd, e,
                            cgm.getBuiltinLibFunction(fd, builtinId));

  // If this is a predefined lib function (e.g. malloc), emit the call
  // using exactly the normal call path.
  if (getContext().BuiltinInfo.isPredefinedLibFunction(builtinId))
    return buildLibraryCall(*this, fd, e,
                            buildScalarExpr(e->getCallee()).getDefiningOp());

  // Check that a call to a target specific builtin has the correct target
  // features.
  // This is down here to avoid non-target specific builtins, however, if
  // generic builtins start to require generic target features then we
  // can move this up to the beginning of the function.
  //   checkTargetFeatures(E, FD);

  if (unsigned vectorWidth =
          getContext().BuiltinInfo.getRequiredVectorWidth(builtinId))
    llvm_unreachable("NYI");

  // See if we have a target specific intrinsic.
  auto name = getContext().BuiltinInfo.getName(builtinId).str();
  Intrinsic::ID intrinsicId = Intrinsic::not_intrinsic;
  StringRef prefix =
      llvm::Triple::getArchTypePrefix(getTarget().getTriple().getArch());
  if (!prefix.empty()) {
    intrinsicId = Intrinsic::getIntrinsicForClangBuiltin(prefix.data(), name);
    // NOTE we don't need to perform a compatibility flag check here since the
    // intrinsics are declared in Builtins*.def via LANGBUILTIN which filter the
    // MS builtins via ALL_MS_LANGUAGES and are filtered earlier.
    if (intrinsicId == Intrinsic::not_intrinsic)
      intrinsicId = Intrinsic::getIntrinsicForMSBuiltin(prefix.data(), name);
  }

  if (intrinsicId != Intrinsic::not_intrinsic) {
    llvm_unreachable("NYI");
  }

  // Some target-specific builtins can have aggregate return values, e.g.
  // __builtin_arm_mve_vld2q_u32. So if the result is an aggregate, force
  // ReturnValue to be non-null, so that the target-specific emission code can
  // always just emit into it.
  TypeEvaluationKind evalKind = getEvaluationKind(e->getType());
  if (evalKind == TEK_Aggregate && returnValue.isNull()) {
    llvm_unreachable("NYI");
  }

  // Now see if we can emit a target-specific builtin.
  if (auto v = buildTargetBuiltinExpr(builtinId, e, returnValue)) {
    switch (evalKind) {
    case TEK_Scalar:
      if (mlir::isa<mlir::cir::VoidType>(v.getType()))
        return RValue::get(nullptr);
      return RValue::get(v);
    case TEK_Aggregate:
      llvm_unreachable("NYI");
    case TEK_Complex:
      llvm_unreachable("No current target builtin returns complex");
    }
    llvm_unreachable("Bad evaluation kind in EmitBuiltinExpr");
  }

  cgm.ErrorUnsupported(e, "builtin function");

  // Unknown builtin, for now just dump it out and return undef.
  return GetUndefRValue(e->getType());
}

mlir::Value CIRGenFunction::buildCheckedArgForBuiltin(const Expr *e,
                                                      BuiltinCheckKind kind) {
  assert((kind == BCK_CLZPassedZero || kind == BCK_CTZPassedZero) &&
         "Unsupported builtin check kind");

  auto value = buildScalarExpr(e);
  if (!SanOpts.has(SanitizerKind::Builtin))
    return value;

  assert(!MissingFeatures::sanitizerBuiltin());
  llvm_unreachable("NYI");
}

static mlir::Value buildTargetArchBuiltinExpr(CIRGenFunction *cgf,
                                              unsigned builtinId,
                                              const CallExpr *e,
                                              ReturnValueSlot returnValue,
                                              llvm::Triple::ArchType arch) {
  // When compiling in HipStdPar mode we have to be conservative in rejecting
  // target specific features in the FE, and defer the possible error to the
  // AcceleratorCodeSelection pass, wherein iff an unsupported target builtin is
  // referenced by an accelerator executable function, we emit an error.
  // Returning nullptr here leads to the builtin being handled in
  // EmitStdParUnsupportedBuiltin.
  if (cgf->getLangOpts().HIPStdPar && cgf->getLangOpts().CUDAIsDevice &&
      arch != cgf->getTarget().getTriple().getArch())
    return nullptr;

  switch (arch) {
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    llvm_unreachable("NYI");
  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_32:
  case llvm::Triple::aarch64_be:
    return cgf->buildAArch64BuiltinExpr(builtinId, e, returnValue, arch);
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
    llvm_unreachable("NYI");
  case llvm::Triple::x86:
  case llvm::Triple::x86_64:
    return cgf->buildX86BuiltinExpr(builtinId, e);
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
    llvm_unreachable("NYI");
  case llvm::Triple::r600:
  case llvm::Triple::amdgcn:
    llvm_unreachable("NYI");
  case llvm::Triple::systemz:
    llvm_unreachable("NYI");
  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    llvm_unreachable("NYI");
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
    llvm_unreachable("NYI");
  case llvm::Triple::hexagon:
    llvm_unreachable("NYI");
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
    llvm_unreachable("NYI");
  default:
    return {};
  }
}

mlir::Value
CIRGenFunction::buildTargetBuiltinExpr(unsigned builtinId, const CallExpr *e,
                                       ReturnValueSlot returnValue) {
  if (getContext().BuiltinInfo.isAuxBuiltinID(builtinId)) {
    assert(getContext().getAuxTargetInfo() && "Missing aux target info");
    return buildTargetArchBuiltinExpr(
        this, getContext().BuiltinInfo.getAuxBuiltinID(builtinId), e,
        returnValue, getContext().getAuxTargetInfo()->getTriple().getArch());
  }

  return buildTargetArchBuiltinExpr(this, builtinId, e, returnValue,
                                    getTarget().getTriple().getArch());
}

void CIRGenFunction::buildVAStartEnd(mlir::Value argValue, bool isStart) {
  // LLVM codegen casts to *i8, no real gain on doing this for CIRGen this
  // early, defer to LLVM lowering.
  if (isStart)
    builder.create<mlir::cir::VAStartOp>(argValue.getLoc(), argValue);
  else
    builder.create<mlir::cir::VAEndOp>(argValue.getLoc(), argValue);
}

/// Checks if using the result of __builtin_object_size(p, @p From) in place of
/// __builtin_object_size(p, @p To) is correct
static bool areBOSTypesCompatible(int from, int to) {
  // Note: Our __builtin_object_size implementation currently treats Type=0 and
  // Type=2 identically. Encoding this implementation detail here may make
  // improving __builtin_object_size difficult in the future, so it's omitted.
  return from == to || (from == 0 && to == 1) || (from == 3 && to == 2);
}

/// Returns a Value corresponding to the size of the given expression.
/// This Value may be either of the following:
///
///   - Reference an argument if `pass_object_size` is used.
///   - A call to a `cir.objsize`.
///
/// EmittedE is the result of emitting `E` as a scalar expr. If it's non-null
/// and we wouldn't otherwise try to reference a pass_object_size parameter,
/// we'll call `cir.objsize` on EmittedE, rather than emitting E.
mlir::Value CIRGenFunction::emitBuiltinObjectSize(const Expr *e, unsigned type,
                                                  mlir::cir::IntType resType,
                                                  mlir::Value emittedE,
                                                  bool isDynamic) {
  // We need to reference an argument if the pointer is a parameter with the
  // pass_object_size attribute.
  if (auto *d = dyn_cast<DeclRefExpr>(e->IgnoreParenImpCasts())) {
    auto *param = dyn_cast<ParmVarDecl>(d->getDecl());
    auto *ps = d->getDecl()->getAttr<PassObjectSizeAttr>();
    if (param != nullptr && ps != nullptr &&
        areBOSTypesCompatible(ps->getType(), type)) {
      auto iter = SizeArguments.find(param);
      assert(iter != SizeArguments.end());

      const ImplicitParamDecl *d = iter->second;
      auto dIter = LocalDeclMap.find(d);
      assert(dIter != LocalDeclMap.end());

      return buildLoadOfScalar(dIter->second, /*isVolatile=*/false,
                               getContext().getSizeType(), e->getBeginLoc());
    }
  }

  // LLVM can't handle Type=3 appropriately, and __builtin_object_size shouldn't
  // evaluate E for side-effects. In either case, just like original LLVM
  // lowering, we shouldn't lower to `cir.objsize`.
  if (type == 3 || (!emittedE && e->HasSideEffects(getContext())))
    llvm_unreachable("NYI");

  auto ptr = emittedE ? emittedE : buildScalarExpr(e);
  assert(mlir::isa<mlir::cir::PointerType>(ptr.getType()) &&
         "Non-pointer passed to __builtin_object_size?");

  // LLVM intrinsics (which CIR lowers to at some point, only supports 0
  // and 2, account for that right now.
  mlir::cir::SizeInfoType sizeInfoTy = ((type & 2) != 0)
                                           ? mlir::cir::SizeInfoType::min
                                           : mlir::cir::SizeInfoType::max;
  // TODO(cir): Heads up for LLVM lowering, For GCC compatibility,
  // __builtin_object_size treat NULL as unknown size.
  return builder.create<mlir::cir::ObjSizeOp>(
      getLoc(e->getSourceRange()), resType, ptr, sizeInfoTy, isDynamic);
}

mlir::Value CIRGenFunction::evaluateOrEmitBuiltinObjectSize(
    const Expr *e, unsigned type, mlir::cir::IntType resType,
    mlir::Value emittedE, bool isDynamic) {
  uint64_t objectSize;
  if (!e->tryEvaluateObjectSize(objectSize, getContext(), type))
    return emitBuiltinObjectSize(e, type, resType, emittedE, isDynamic);
  return builder.getConstInt(getLoc(e->getSourceRange()), resType, objectSize);
}

/// Given a builtin id for a function like "__builtin_fabsf", return a Function*
/// for "fabsf".
mlir::cir::FuncOp CIRGenModule::getBuiltinLibFunction(const FunctionDecl *fd,
                                                      unsigned builtinId) {
  assert(astCtx.BuiltinInfo.isLibFunction(builtinId));

  // Get the name, skip over the __builtin_ prefix (if necessary).
  StringRef name;
  GlobalDecl d(fd);

  // TODO: This list should be expanded or refactored after all GCC-compatible
  // std libcall builtins are implemented.
  static SmallDenseMap<unsigned, StringRef, 64> f128Builtins{
      {Builtin::BI__builtin___fprintf_chk, "__fprintf_chkieee128"},
      {Builtin::BI__builtin___printf_chk, "__printf_chkieee128"},
      {Builtin::BI__builtin___snprintf_chk, "__snprintf_chkieee128"},
      {Builtin::BI__builtin___sprintf_chk, "__sprintf_chkieee128"},
      {Builtin::BI__builtin___vfprintf_chk, "__vfprintf_chkieee128"},
      {Builtin::BI__builtin___vprintf_chk, "__vprintf_chkieee128"},
      {Builtin::BI__builtin___vsnprintf_chk, "__vsnprintf_chkieee128"},
      {Builtin::BI__builtin___vsprintf_chk, "__vsprintf_chkieee128"},
      {Builtin::BI__builtin_fprintf, "__fprintfieee128"},
      {Builtin::BI__builtin_printf, "__printfieee128"},
      {Builtin::BI__builtin_snprintf, "__snprintfieee128"},
      {Builtin::BI__builtin_sprintf, "__sprintfieee128"},
      {Builtin::BI__builtin_vfprintf, "__vfprintfieee128"},
      {Builtin::BI__builtin_vprintf, "__vprintfieee128"},
      {Builtin::BI__builtin_vsnprintf, "__vsnprintfieee128"},
      {Builtin::BI__builtin_vsprintf, "__vsprintfieee128"},
      {Builtin::BI__builtin_fscanf, "__fscanfieee128"},
      {Builtin::BI__builtin_scanf, "__scanfieee128"},
      {Builtin::BI__builtin_sscanf, "__sscanfieee128"},
      {Builtin::BI__builtin_vfscanf, "__vfscanfieee128"},
      {Builtin::BI__builtin_vscanf, "__vscanfieee128"},
      {Builtin::BI__builtin_vsscanf, "__vsscanfieee128"},
      {Builtin::BI__builtin_nexttowardf128, "__nexttowardieee128"},
  };

  // The AIX library functions frexpl, ldexpl, and modfl are for 128-bit
  // IBM 'long double' (i.e. __ibm128). Map to the 'double' versions
  // if it is 64-bit 'long double' mode.
  static SmallDenseMap<unsigned, StringRef, 4> aixLongDouble64Builtins{
      {Builtin::BI__builtin_frexpl, "frexp"},
      {Builtin::BI__builtin_ldexpl, "ldexp"},
      {Builtin::BI__builtin_modfl, "modf"},
  };

  // If the builtin has been declared explicitly with an assembler label,
  // use the mangled name. This differs from the plain label on platforms
  // that prefix labels.
  if (fd->hasAttr<AsmLabelAttr>())
    name = getMangledName(d);
  else {
    // TODO: This mutation should also be applied to other targets other than
    // PPC, after backend supports IEEE 128-bit style libcalls.
    if (getTriple().isPPC64() &&
        &getTarget().getLongDoubleFormat() == &llvm::APFloat::IEEEquad() &&
        f128Builtins.find(builtinId) != f128Builtins.end())
      name = f128Builtins[builtinId];
    else if (getTriple().isOSAIX() &&
             &getTarget().getLongDoubleFormat() ==
                 &llvm::APFloat::IEEEdouble() &&
             aixLongDouble64Builtins.find(builtinId) !=
                 aixLongDouble64Builtins.end())
      name = aixLongDouble64Builtins[builtinId];
    else
      name = astCtx.BuiltinInfo.getName(builtinId).substr(10);
  }

  auto ty = getTypes().ConvertType(fd->getType());
  return GetOrCreateCIRFunction(name, ty, d, /*ForVTable=*/false);
}
