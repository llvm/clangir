//===--- CIRGenExprScalar.cpp - Emit CIR Code for Scalar Exprs ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Expr nodes with scalar CIR types as CIR code.
//
//===----------------------------------------------------------------------===//

#include "Address.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenOpenMPRuntime.h"
#include "TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

namespace {

struct BinOpInfo {
  mlir::Value lhs;
  mlir::Value rhs;
  SourceRange loc;
  QualType fullType;             // Type of operands and result
  QualType compType;             // Type used for computations. Element type
                                 // for vectors, otherwise same as FullType.
  BinaryOperator::Opcode opcode; // Opcode of BinOp to perform
  FPOptions fpFeatures;
  const Expr *e; // Entire expr, for error unsupported.  May not be binop.

  /// Check if the binop computes a division or a remainder.
  bool isDivremOp() const {
    return opcode == BO_Div || opcode == BO_Rem || opcode == BO_DivAssign ||
           opcode == BO_RemAssign;
  }

  /// Check if the binop can result in integer overflow.
  bool mayHaveIntegerOverflow() const {
    // Without constant input, we can't rule out overflow.
    auto lhsci = dyn_cast<mlir::cir::ConstantOp>(lhs.getDefiningOp());
    auto rhsci = dyn_cast<mlir::cir::ConstantOp>(rhs.getDefiningOp());
    if (!lhsci || !rhsci)
      return true;

    llvm::APInt result;
    assert(!MissingFeatures::mayHaveIntegerOverflow());
    llvm_unreachable("NYI");
    return false;
  }

  /// Check if at least one operand is a fixed point type. In such cases,
  /// this operation did not follow usual arithmetic conversion and both
  /// operands might not be of the same type.
  bool isFixedPointOp() const {
    // We cannot simply check the result type since comparison operations
    // return an int.
    if (const auto *binOp = llvm::dyn_cast<BinaryOperator>(e)) {
      QualType lhsType = binOp->getLHS()->getType();
      QualType rhsType = binOp->getRHS()->getType();
      return lhsType->isFixedPointType() || rhsType->isFixedPointType();
    }
    if (const auto *unOp = llvm::dyn_cast<UnaryOperator>(e))
      return unOp->getSubExpr()->getType()->isFixedPointType();
    return false;
  }
};

static bool promotionIsPotentiallyEligibleForImplicitIntegerConversionCheck(
    QualType srcType, QualType dstType) {
  return srcType->isIntegerType() && dstType->isIntegerType();
}

class ScalarExprEmitter : public StmtVisitor<ScalarExprEmitter, mlir::Value> {
  CIRGenFunction &CGF;
  CIRGenBuilderTy &Builder;
  bool ignoreResultAssign;

public:
  ScalarExprEmitter(CIRGenFunction &cgf, CIRGenBuilderTy &builder,
                    bool ira = false)
      : CGF(cgf), Builder(builder), ignoreResultAssign(ira) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  bool testAndClearIgnoreResultAssign() {
    bool i = ignoreResultAssign;
    ignoreResultAssign = false;
    return i;
  }

  mlir::Type convertType(QualType t) { return CGF.ConvertType(t); }
  LValue buildLValue(const Expr *e) { return CGF.buildLValue(e); }
  LValue buildCheckedLValue(const Expr *e, CIRGenFunction::TypeCheckKind tck) {
    return CGF.buildCheckedLValue(e, tck);
  }

  mlir::Value buildComplexToScalarConversion(mlir::Location loc, mlir::Value v,
                                             CastKind kind, QualType destTy);

  /// Emit a value that corresponds to null for the given type.
  mlir::Value buildNullValue(QualType ty, mlir::Location loc);

  mlir::Value buildPromotedValue(mlir::Value result, QualType promotionType) {
    return Builder.createFloatingCast(result, convertType(promotionType));
  }

  mlir::Value buildUnPromotedValue(mlir::Value result, QualType exprType) {
    return Builder.createFloatingCast(result, convertType(exprType));
  }

  mlir::Value buildPromoted(const Expr *e, QualType promotionType);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *e) {
    return StmtVisitor<ScalarExprEmitter, mlir::Value>::Visit(e);
  }

  mlir::Value VisitStmt(Stmt *s) {
    s->dump(llvm::errs(), CGF.getContext());
    llvm_unreachable("Stmt can't have complex result type!");
  }

  mlir::Value VisitExpr(Expr *e) {
    // Crashing here for "ScalarExprClassName"? Please implement
    // VisitScalarExprClassName(...) to get this working.
    emitError(CGF.getLoc(e->getExprLoc()), "scalar exp no implemented: '")
        << e->getStmtClassName() << "'";
    llvm_unreachable("NYI");
    return {};
  }

  mlir::Value VisitConstantExpr(ConstantExpr *e) { llvm_unreachable("NYI"); }
  mlir::Value VisitParenExpr(ParenExpr *pe) { return Visit(pe->getSubExpr()); }
  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *e) {
    return Visit(e->getReplacement());
  }
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *ge) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCoawaitExpr(CoawaitExpr *s) {
    return CGF.buildCoawaitExpr(*s).getScalarVal();
  }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *s) {
    return CGF.buildCoyieldExpr(*s).getScalarVal();
  }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *e) {
    llvm_unreachable("NYI");
  }

  // Leaves.
  mlir::Value VisitIntegerLiteral(const IntegerLiteral *e) {
    mlir::Type ty = CGF.getCIRType(e->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(e->getExprLoc()), ty,
        Builder.getAttr<mlir::cir::IntAttr>(ty, e->getValue()));
  }

  mlir::Value VisitFixedPointLiteral(const FixedPointLiteral *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitFloatingLiteral(const FloatingLiteral *e) {
    mlir::Type ty = CGF.getCIRType(e->getType());
    assert(mlir::isa<mlir::cir::CIRFPTypeInterface>(ty) &&
           "expect floating-point type");
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(e->getExprLoc()), ty,
        Builder.getAttr<mlir::cir::FPAttr>(ty, e->getValue()));
  }
  mlir::Value VisitCharacterLiteral(const CharacterLiteral *e) {
    mlir::Type ty = CGF.getCIRType(e->getType());
    auto loc = CGF.getLoc(e->getExprLoc());
    auto init = mlir::cir::IntAttr::get(ty, e->getValue());
    return Builder.create<mlir::cir::ConstantOp>(loc, ty, init);
  }
  mlir::Value VisitObjCBoolLiteralExpr(const ObjCBoolLiteralExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr *e) {
    mlir::Type ty = CGF.getCIRType(e->getType());
    return Builder.create<mlir::cir::ConstantOp>(
        CGF.getLoc(e->getExprLoc()), ty, Builder.getCIRBoolAttr(e->getValue()));
  }

  mlir::Value VisitCXXScalarValueInitExpr(const CXXScalarValueInitExpr *e) {
    if (e->getType()->isVoidType())
      return nullptr;

    return buildNullValue(e->getType(), CGF.getLoc(e->getSourceRange()));
  }
  mlir::Value VisitGNUNullExpr(const GNUNullExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitOffsetOfExpr(OffsetOfExpr *e) {
    // Try folding the offsetof to a constant.
    Expr::EvalResult evResult;
    if (e->EvaluateAsInt(evResult, CGF.getContext())) {
      llvm::APSInt value = evResult.Val.getInt();
      return Builder.getConstInt(CGF.getLoc(e->getExprLoc()), value);
    }

    llvm_unreachable("NYI");
  }

  mlir::Value VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *e);
  mlir::Value VisitAddrLabelExpr(const AddrLabelExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSizeOfPackExpr(SizeOfPackExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitPseudoObjectExpr(PseudoObjectExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSYCLUniqueStableNameExpr(SYCLUniqueStableNameExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *e) {
    if (e->isGLValue())
      llvm_unreachable("NYI");

    // Otherwise, assume the mapping is the scalar directly.
    return CGF.getOrCreateOpaqueRValueMapping(e).getScalarVal();
  }

  /// Emits the address of the l-value, then loads and returns the result.
  mlir::Value buildLoadOfLValue(const Expr *e) {
    LValue lv = CGF.buildLValue(e);
    // FIXME: add some akin to EmitLValueAlignmentAssumption(E, V);
    return CGF.buildLoadOfLValue(lv, e->getExprLoc()).getScalarVal();
  }

  mlir::Value buildLoadOfLValue(LValue lv, SourceLocation loc) {
    return CGF.buildLoadOfLValue(lv, loc).getScalarVal();
  }

  // l-values
  mlir::Value VisitDeclRefExpr(DeclRefExpr *e) {
    if (CIRGenFunction::ConstantEmission constant = CGF.tryEmitAsConstant(e)) {
      return CGF.buildScalarConstant(constant, e);
    }
    return buildLoadOfLValue(e);
  }

  mlir::Value VisitObjCSelectorExpr(ObjCSelectorExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCProtocolExpr(ObjCProtocolExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value visitObjCiVarRefExpr(ObjCIvarRefExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCMessageExpr(ObjCMessageExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCIsaExpr(ObjCIsaExpr *e) { llvm_unreachable("NYI"); }
  mlir::Value VisitObjCAvailabilityCheckExpr(ObjCAvailabilityCheckExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArraySubscriptExpr(ArraySubscriptExpr *e) {
    // Do we need anything like TestAndClearIgnoreResultAssign()?

    if (e->getBase()->getType()->isVectorType()) {
      assert(!MissingFeatures::scalableVectors() &&
             "NYI: index into scalable vector");
      // Subscript of vector type.  This is handled differently, with a custom
      // operation.
      mlir::Value vecValue = Visit(e->getBase());
      mlir::Value indexValue = Visit(e->getIdx());
      return CGF.builder.create<mlir::cir::VecExtractOp>(
          CGF.getLoc(e->getSourceRange()), vecValue, indexValue);
    }

    // Just load the lvalue formed by the subscript expression.
    return buildLoadOfLValue(e);
  }

  mlir::Value VisitMatrixSubscriptExpr(MatrixSubscriptExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
    if (E->getNumSubExprs() == 2) {
      // The undocumented form of __builtin_shufflevector.
      mlir::Value inputVec = Visit(E->getExpr(0));
      mlir::Value indexVec = Visit(E->getExpr(1));
      return CGF.builder.create<mlir::cir::VecShuffleDynamicOp>(
          CGF.getLoc(E->getSourceRange()), inputVec, indexVec);
    } // The documented form of __builtin_shufflevector, where the indices are
      // a variable number of integer constants. The constants will be stored
      // in an ArrayAttr.
      mlir::Value Vec1 = Visit(E->getExpr(0));
      mlir::Value Vec2 = Visit(E->getExpr(1));
      SmallVector<mlir::Attribute, 8> Indices;
      for (unsigned i = 2; i < E->getNumSubExprs(); ++i) {
        Indices.push_back(mlir::cir::IntAttr::get(
            CGF.builder.getSInt64Ty(),
            E->getExpr(i)
                ->EvaluateKnownConstInt(CGF.getContext())
                .getSExtValue()));
      }
      return CGF.builder.create<mlir::cir::VecShuffleOp>(
          CGF.getLoc(E->getSourceRange()), CGF.getCIRType(E->getType()), Vec1,
          Vec2, CGF.builder.getArrayAttr(Indices));
  }
  mlir::Value VisitConvertVectorExpr(ConvertVectorExpr *e) {
    // __builtin_convertvector is an element-wise cast, and is implemented as a
    // regular cast. The back end handles casts of vectors correctly.
    return buildScalarConversion(Visit(e->getSrcExpr()),
                                 e->getSrcExpr()->getType(), e->getType(),
                                 e->getSourceRange().getBegin());
  }

  mlir::Value VisitExtVectorElementExpr(Expr *e) {
    return buildLoadOfLValue(e);
  }

  mlir::Value VisitMemberExpr(MemberExpr *e);
  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
    return buildLoadOfLValue(e);
  }

  mlir::Value VisitInitListExpr(InitListExpr *e);

  mlir::Value VisitArrayInitIndexExpr(ArrayInitIndexExpr *e) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitImplicitValueInitExpr(const ImplicitValueInitExpr *e) {
    return buildNullValue(e->getType(), CGF.getLoc(e->getSourceRange()));
  }
  mlir::Value VisitExplicitCastExpr(ExplicitCastExpr *e) {
    return VisitCastExpr(e);
  }
  mlir::Value VisitCastExpr(CastExpr *e);
  mlir::Value VisitCallExpr(const CallExpr *e);

  mlir::Value VisitStmtExpr(StmtExpr *e) {
    assert(!MissingFeatures::stmtExprEvaluation() && "NYI");
    Address retAlloca =
        CGF.buildCompoundStmt(*e->getSubStmt(), !e->getType()->isVoidType());
    if (!retAlloca.isValid())
      return {};

    // FIXME(cir): This is a work around the ScopeOp builder. If we build the
    // ScopeOp before its body, we would be able to create the retAlloca
    // direclty in the parent scope removing the need to hoist it.
    assert(retAlloca.getDefiningOp() && "expected a alloca op");
    CGF.getBuilder().hoistAllocaToParentRegion(
        cast<mlir::cir::AllocaOp>(retAlloca.getDefiningOp()));

    return CGF.buildLoadOfScalar(CGF.makeAddrLValue(retAlloca, e->getType()),
                                 e->getExprLoc());
  }

  // Unary Operators.
  mlir::Value VisitUnaryPostDec(const UnaryOperator *e) {
    LValue lv = buildLValue(e->getSubExpr());
    return buildScalarPrePostIncDec(e, lv, false, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *e) {
    LValue lv = buildLValue(e->getSubExpr());
    return buildScalarPrePostIncDec(e, lv, true, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *e) {
    LValue lv = buildLValue(e->getSubExpr());
    return buildScalarPrePostIncDec(e, lv, false, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *e) {
    LValue lv = buildLValue(e->getSubExpr());
    return buildScalarPrePostIncDec(e, lv, true, true);
  }
  mlir::Value buildScalarPrePostIncDec(const UnaryOperator *e, LValue lv,
                                       bool isInc, bool isPre) {
    assert(!CGF.getLangOpts().OpenMP && "Not implemented");
    QualType type = e->getSubExpr()->getType();

    int amount = (isInc ? 1 : -1);
    bool atomicPHI = false;
    mlir::Value value{};
    mlir::Value input{};

    if (const AtomicType *atomicTy = type->getAs<AtomicType>()) {
      llvm_unreachable("no atomics inc/dec yet");
    } else {
      value = buildLoadOfLValue(lv, e->getExprLoc());
      input = value;
    }

    // NOTE: When possible, more frequent cases are handled first.

    // Special case of integer increment that we have to check first: bool++.
    // Due to promotion rules, we get:
    //   bool++ -> bool = bool + 1
    //          -> bool = (int)bool + 1
    //          -> bool = ((int)bool + 1 != 0)
    // An interesting aspect of this is that increment is always true.
    // Decrement does not have this property.
    if (isInc && type->isBooleanType()) {
      value = Builder.create<mlir::cir::ConstantOp>(
          CGF.getLoc(e->getExprLoc()), CGF.getCIRType(type),
          Builder.getCIRBoolAttr(true));
    } else if (type->isIntegerType()) {
      QualType promotedType;
      bool canPerformLossyDemotionCheck = false;
      if (CGF.getContext().isPromotableIntegerType(type)) {
        promotedType = CGF.getContext().getPromotedIntegerType(type);
        assert(promotedType != type && "Shouldn't promote to the same type.");
        canPerformLossyDemotionCheck = true;
        canPerformLossyDemotionCheck &=
            CGF.getContext().getCanonicalType(type) !=
            CGF.getContext().getCanonicalType(promotedType);
        canPerformLossyDemotionCheck &=
            promotionIsPotentiallyEligibleForImplicitIntegerConversionCheck(
                type, promotedType);

        // TODO(cir): Currently, we store bitwidths in CIR types only for
        // integers. This might also be required for other types.
        auto srcCirTy = mlir::dyn_cast<mlir::cir::IntType>(convertType(type));
        auto promotedCirTy =
            mlir::dyn_cast<mlir::cir::IntType>(convertType(type));
        assert(srcCirTy && promotedCirTy && "Expected integer type");

        assert(
            (!canPerformLossyDemotionCheck ||
             type->isSignedIntegerOrEnumerationType() ||
             promotedType->isSignedIntegerOrEnumerationType() ||
             srcCirTy.getWidth() == promotedCirTy.getWidth()) &&
            "The following check expects that if we do promotion to different "
            "underlying canonical type, at least one of the types (either "
            "base or promoted) will be signed, or the bitwidths will match.");
      }

      if (CGF.SanOpts.hasOneOf(
              SanitizerKind::ImplicitIntegerArithmeticValueChange) &&
          canPerformLossyDemotionCheck) {
        llvm_unreachable(
            "perform lossy demotion case for inc/dec not implemented yet");
      } else if (e->canOverflow() && type->isSignedIntegerOrEnumerationType()) {
        value = buildIncDecConsiderOverflowBehavior(e, value, isInc);
      } else if (e->canOverflow() && type->isUnsignedIntegerType() &&
                 CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow)) {
        llvm_unreachable(
            "unsigned integer overflow sanitized inc/dec not implemented");
      } else {
        auto kind = e->isIncrementOp() ? mlir::cir::UnaryOpKind::Inc
                                       : mlir::cir::UnaryOpKind::Dec;
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        value = buildUnaryOp(e, kind, input);
      }
      // Next most common: pointer increment.
    } else if (const PointerType *ptr = type->getAs<PointerType>()) {
      QualType type = ptr->getPointeeType();
      if (const VariableArrayType *vla =
              CGF.getContext().getAsVariableArrayType(type)) {
        // VLA types don't have constant size.
        llvm_unreachable("NYI");
      } else if (type->isFunctionType()) {
        // Arithmetic on function pointers (!) is just +-1.
        llvm_unreachable("NYI");
      } else {
        // For everything else, we can just do a simple increment.
        auto loc = CGF.getLoc(e->getSourceRange());
        auto &builder = CGF.getBuilder();
        auto amt = builder.getSInt32(amount, loc);
        if (CGF.getLangOpts().isSignedOverflowDefined()) {
          value = builder.create<mlir::cir::PtrStrideOp>(loc, value.getType(),
                                                         value, amt);
        } else {
          value = builder.create<mlir::cir::PtrStrideOp>(loc, value.getType(),
                                                         value, amt);
          assert(!MissingFeatures::emitCheckedInBoundsGEP());
        }
      }
    } else if (type->isVectorType()) {
      llvm_unreachable("no vector inc/dec yet");
    } else if (type->isRealFloatingType()) {
      // TODO(cir): CGFPOptionsRAII
      assert(!MissingFeatures::cgFPOptionsRAII());

      if (type->isHalfType() && !CGF.getContext().getLangOpts().NativeHalfType)
        llvm_unreachable("__fp16 type NYI");

      if (mlir::isa<mlir::cir::SingleType, mlir::cir::DoubleType>(
              value.getType())) {
        // Create the inc/dec operation.
        // NOTE(CIR): clang calls CreateAdd but folds this to a unary op
        auto kind =
            (isInc ? mlir::cir::UnaryOpKind::Inc : mlir::cir::UnaryOpKind::Dec);
        value = buildUnaryOp(e, kind, input);
      } else {
        // Remaining types are Half, Bfloat16, LongDouble, __ibm128 or
        // __float128. Convert from float.

        llvm::APFloat f(static_cast<float>(amount));
        bool ignored;
        const llvm::fltSemantics *fs;
        // Don't use getFloatTypeSemantics because Half isn't
        // necessarily represented using the "half" LLVM type.
        if (mlir::isa<mlir::cir::LongDoubleType>(value.getType()))
          fs = &CGF.getTarget().getLongDoubleFormat();
        else if (mlir::isa<mlir::cir::FP16Type>(value.getType()))
          fs = &CGF.getTarget().getHalfFormat();
        else if (mlir::isa<mlir::cir::BF16Type>(value.getType()))
          fs = &CGF.getTarget().getBFloat16Format();
        else
          llvm_unreachable("fp128 / ppc_fp128 NYI");
        f.convert(*fs, llvm::APFloat::rmTowardZero, &ignored);

        auto loc = CGF.getLoc(e->getExprLoc());
        auto amt = Builder.getConstant(
            loc, mlir::cir::FPAttr::get(value.getType(), f));
        value = Builder.createBinop(value, mlir::cir::BinOpKind::Add, amt);
      }

      if (type->isHalfType() && !CGF.getContext().getLangOpts().NativeHalfType)
        llvm_unreachable("NYI");

    } else if (type->isFixedPointType()) {
      llvm_unreachable("no fixed point inc/dec yet");
    } else {
      assert(type->castAs<ObjCObjectPointerType>());
      llvm_unreachable("no objc pointer type inc/dec yet");
    }

    if (atomicPHI) {
      llvm_unreachable("NYI");
    }

    CIRGenFunction::SourceLocRAIIObject sourceloc{
        CGF, CGF.getLoc(e->getSourceRange())};

    // Store the updated result through the lvalue
    if (lv.isBitField())
      CGF.buildStoreThroughBitfieldLValue(RValue::get(value), lv, value);
    else
      CGF.buildStoreThroughLValue(RValue::get(value), lv);

    // If this is a postinc, return the value read from memory, otherwise use
    // the updated value.
    return isPre ? value : input;
  }

  mlir::Value buildIncDecConsiderOverflowBehavior(const UnaryOperator *e,
                                                  mlir::Value inVal,
                                                  bool isInc) {
    // NOTE(CIR): The SignedOverflowBehavior is attached to the global ModuleOp
    // and the nsw behavior is handled during lowering.
    auto kind = e->isIncrementOp() ? mlir::cir::UnaryOpKind::Inc
                                   : mlir::cir::UnaryOpKind::Dec;
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      return buildUnaryOp(e, kind, inVal);
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return buildUnaryOp(e, kind, inVal);
      llvm_unreachable(
          "inc/dec overflow behavior SOB_Undefined not implemented yet");
      break;
    case LangOptions::SOB_Trapping:
      if (!e->canOverflow())
        return buildUnaryOp(e, kind, inVal);
      llvm_unreachable(
          "inc/dec overflow behavior SOB_Trapping not implemented yet");
      break;
    }
  }

  mlir::Value VisitUnaryAddrOf(const UnaryOperator *e) {
    if (llvm::isa<MemberPointerType>(e->getType()))
      return CGF.cgm.buildMemberPointerConstant(e);

    return CGF.buildLValue(e->getSubExpr()).getPointer();
  }

  mlir::Value VisitUnaryDeref(const UnaryOperator *e) {
    if (e->getType()->isVoidType())
      return Visit(e->getSubExpr()); // the actual value should be unused
    return buildLoadOfLValue(e);
  }
  mlir::Value VisitUnaryPlus(const UnaryOperator *e,
                             QualType promotionType = QualType()) {
    QualType promotionTy = promotionType.isNull()
                               ? getPromotionType(e->getSubExpr()->getType())
                               : promotionType;
    auto result = visitPlus(e, promotionTy);
    if (result && !promotionTy.isNull())
      return buildUnPromotedValue(result, e->getType());
    return result;
  }

  mlir::Value visitPlus(const UnaryOperator *e,
                        QualType promotionType = QualType()) {
    // This differs from gcc, though, most likely due to a bug in gcc.
    testAndClearIgnoreResultAssign();

    mlir::Value operand;
    if (!promotionType.isNull())
      operand = CGF.buildPromotedScalarExpr(e->getSubExpr(), promotionType);
    else
      operand = Visit(e->getSubExpr());

    return buildUnaryOp(e, mlir::cir::UnaryOpKind::Plus, operand);
  }

  mlir::Value VisitUnaryMinus(const UnaryOperator *e,
                              QualType promotionType = QualType()) {
    QualType promotionTy = promotionType.isNull()
                               ? getPromotionType(e->getSubExpr()->getType())
                               : promotionType;
    auto result = visitMinus(e, promotionTy);
    if (result && !promotionTy.isNull())
      return buildUnPromotedValue(result, e->getType());
    return result;
  }

  mlir::Value visitMinus(const UnaryOperator *e, QualType promotionType) {
    testAndClearIgnoreResultAssign();

    mlir::Value operand;
    if (!promotionType.isNull())
      operand = CGF.buildPromotedScalarExpr(e->getSubExpr(), promotionType);
    else
      operand = Visit(e->getSubExpr());

    // NOTE: LLVM codegen will lower this directly to either a FNeg
    // or a Sub instruction.  In CIR this will be handled later in LowerToLLVM.
    return buildUnaryOp(e, mlir::cir::UnaryOpKind::Minus, operand);
  }

  mlir::Value VisitUnaryNot(const UnaryOperator *e) {
    testAndClearIgnoreResultAssign();
    mlir::Value op = Visit(e->getSubExpr());
    return buildUnaryOp(e, mlir::cir::UnaryOpKind::Not, op);
  }

  mlir::Value VisitUnaryLNot(const UnaryOperator *e);
  mlir::Value VisitUnaryReal(const UnaryOperator *e) { return visitReal(e); }
  mlir::Value VisitUnaryImag(const UnaryOperator *e) { return visitImag(e); }

  mlir::Value visitReal(const UnaryOperator *e);
  mlir::Value visitImag(const UnaryOperator *e);

  mlir::Value VisitUnaryExtension(const UnaryOperator *e) {
    // __extension__ doesn't requred any codegen
    // just forward the value
    return Visit(e->getSubExpr());
  }

  mlir::Value buildUnaryOp(const UnaryOperator *e, mlir::cir::UnaryOpKind kind,
                           mlir::Value input) {
    return Builder.create<mlir::cir::UnaryOp>(
        CGF.getLoc(e->getSourceRange().getBegin()), input.getType(), kind,
        input);
  }

  // C++
  mlir::Value VisitMaterializeTemporaryExpr(const MaterializeTemporaryExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitSourceLocExpr(SourceLocExpr *e) { llvm_unreachable("NYI"); }
  mlir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *dae) {
    CIRGenFunction::CXXDefaultArgExprScope scope(CGF, dae);
    return Visit(dae->getExpr());
  }
  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    CIRGenFunction::CXXDefaultInitExprScope scope(CGF, die);
    return Visit(die->getExpr());
  }

  mlir::Value VisitCXXThisExpr(CXXThisExpr *te) { return CGF.LoadCXXThis(); }

  mlir::Value VisitExprWithCleanups(ExprWithCleanups *e);
  mlir::Value VisitCXXNewExpr(const CXXNewExpr *e) {
    return CGF.buildCXXNewExpr(e);
  }
  mlir::Value VisitCXXDeleteExpr(const CXXDeleteExpr *e) {
    CGF.buildCXXDeleteExpr(e);
    return {};
  }
  mlir::Value VisitTypeTraitExpr(const TypeTraitExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value
  VisitConceptSpecializationExpr(const ConceptSpecializationExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitRequiresExpr(const RequiresExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArrayTypeTraitExpr(const ArrayTypeTraitExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitExpressionTraitExpr(const ExpressionTraitExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXPseudoDestructorExpr(const CXXPseudoDestructorExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXNullPtrLiteralExpr(CXXNullPtrLiteralExpr *e) {
    return buildNullValue(e->getType(), CGF.getLoc(e->getSourceRange()));
  }
  mlir::Value VisitCXXThrowExpr(CXXThrowExpr *e) {
    CGF.buildCXXThrowExpr(e);
    return nullptr;
  }
  mlir::Value VisitCXXNoexceptExpr(CXXNoexceptExpr *e) {
    llvm_unreachable("NYI");
  }

  /// Perform a pointer to boolean conversion.
  mlir::Value buildPointerToBoolConversion(mlir::Value v, QualType qt) {
    // TODO(cir): comparing the ptr to null is done when lowering CIR to LLVM.
    // We might want to have a separate pass for these types of conversions.
    return CGF.getBuilder().createPtrToBoolCast(v);
  }

  // Comparisons.
#define VISITCOMP(CODE)                                                        \
  mlir::Value VisitBin##CODE(const BinaryOperator *E) { return buildCmp(E); }
  VISITCOMP(LT)
  VISITCOMP(GT)
  VISITCOMP(LE)
  VISITCOMP(GE)
  VISITCOMP(EQ)
  VISITCOMP(NE)
#undef VISITCOMP

  mlir::Value VisitBinAssign(const BinaryOperator *e);
  mlir::Value VisitBinLAnd(const BinaryOperator *b);
  mlir::Value VisitBinLOr(const BinaryOperator *b);
  mlir::Value VisitBinComma(const BinaryOperator *e) {
    CGF.buildIgnoredExpr(e->getLHS());
    // NOTE: We don't need to EnsureInsertPoint() like LLVM codegen.
    return Visit(e->getRHS());
  }

  mlir::Value VisitBinPtrMemD(const BinaryOperator *e) {
    return buildLoadOfLValue(e);
  }

  mlir::Value VisitBinPtrMemI(const BinaryOperator *e) {
    return buildLoadOfLValue(e);
  }

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *e) {
    return Visit(e->getSemanticForm());
  }

  // Other Operators.
  mlir::Value VisitBlockExpr(const BlockExpr *e) { llvm_unreachable("NYI"); }
  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *e);
  mlir::Value VisitChooseExpr(ChooseExpr *e) { llvm_unreachable("NYI"); }
  mlir::Value VisitVAArgExpr(VAArgExpr *ve);
  mlir::Value VisitObjCStringLiteral(const ObjCStringLiteral *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCBoxedExpr(ObjCBoxedExpr *e) { llvm_unreachable("NYI"); }
  mlir::Value VisitObjCArrayLiteral(ObjCArrayLiteral *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCDictionaryLiteral(ObjCDictionaryLiteral *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitAsTypeExpr(AsTypeExpr *e) { llvm_unreachable("NYI"); }
  mlir::Value VisitAtomicExpr(AtomicExpr *e) {
    return CGF.buildAtomicExpr(e).getScalarVal();
  }

  // Emit a conversion from the specified type to the specified destination
  // type, both of which are CIR scalar types.
  struct ScalarConversionOpts {
    bool treatBooleanAsSigned;
    bool emitImplicitIntegerTruncationChecks;
    bool emitImplicitIntegerSignChangeChecks;

    ScalarConversionOpts()
        : treatBooleanAsSigned(false),
          emitImplicitIntegerTruncationChecks(false),
          emitImplicitIntegerSignChangeChecks(false) {}

    ScalarConversionOpts(clang::SanitizerSet sanOpts)
        : treatBooleanAsSigned(false),
          emitImplicitIntegerTruncationChecks(
              sanOpts.hasOneOf(SanitizerKind::ImplicitIntegerTruncation)),
          emitImplicitIntegerSignChangeChecks(
              sanOpts.has(SanitizerKind::ImplicitIntegerSignChange)) {}
  };
  mlir::Value buildScalarCast(mlir::Value src, QualType srcType,
                              QualType dstType, mlir::Type srcTy,
                              mlir::Type dstTy, ScalarConversionOpts opts);

  BinOpInfo buildBinOps(const BinaryOperator *e,
                        QualType promotionType = QualType()) {
    BinOpInfo result;
    result.lhs = CGF.buildPromotedScalarExpr(e->getLHS(), promotionType);
    result.rhs = CGF.buildPromotedScalarExpr(e->getRHS(), promotionType);
    if (!promotionType.isNull())
      result.fullType = promotionType;
    else
      result.fullType = e->getType();
    result.compType = result.fullType;
    if (const auto *vecType = dyn_cast_or_null<VectorType>(result.fullType)) {
      result.compType = vecType->getElementType();
    }
    result.opcode = e->getOpcode();
    result.loc = e->getSourceRange();
    // TODO: Result.FPFeatures
    assert(!MissingFeatures::getFPFeaturesInEffect());
    result.e = e;
    return result;
  }

  mlir::Value buildMul(const BinOpInfo &ops);
  mlir::Value buildDiv(const BinOpInfo &ops);
  mlir::Value buildRem(const BinOpInfo &ops);
  mlir::Value buildAdd(const BinOpInfo &ops);
  mlir::Value buildSub(const BinOpInfo &ops);
  mlir::Value buildShl(const BinOpInfo &ops);
  mlir::Value buildShr(const BinOpInfo &ops);
  mlir::Value buildAnd(const BinOpInfo &ops);
  mlir::Value buildXor(const BinOpInfo &ops);
  mlir::Value buildOr(const BinOpInfo &ops);

  LValue buildCompoundAssignLValue(
      const CompoundAssignOperator *e,
      mlir::Value (ScalarExprEmitter::*f)(const BinOpInfo &),
      mlir::Value &result);
  mlir::Value
  buildCompoundAssign(const CompoundAssignOperator *e,
                      mlir::Value (ScalarExprEmitter::*f)(const BinOpInfo &));

  // TODO(cir): Candidate to be in a common AST helper between CIR and LLVM
  // codegen.
  QualType getPromotionType(QualType ty) {
    if (auto *ct = ty->getAs<ComplexType>()) {
      llvm_unreachable("NYI");
    }
    if (ty.UseExcessPrecision(CGF.getContext())) {
      if (auto *vt = ty->getAs<VectorType>())
        llvm_unreachable("NYI");
      return CGF.getContext().FloatTy;
    }
    return QualType();
  }

  // Binary operators and binary compound assignment operators.
#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) {                          \
    QualType promotionTy = getPromotionType(E->getType());                     \
    auto result = build##OP(buildBinOps(E, promotionTy));                      \
    if (result && !promotionTy.isNull())                                       \
      result = buildUnPromotedValue(result, E->getType());                     \
    return result;                                                             \
  }                                                                            \
  mlir::Value VisitBin##OP##Assign(const CompoundAssignOperator *E) {          \
    return buildCompoundAssign(E, &ScalarExprEmitter::build##OP);              \
  }

  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
  HANDLEBINOP(Rem)
  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
  HANDLEBINOP(Shl)
  HANDLEBINOP(Shr)
  HANDLEBINOP(And)
  HANDLEBINOP(Xor)
  HANDLEBINOP(Or)
#undef HANDLEBINOP

  mlir::Value buildCmp(const BinaryOperator *e) {
    mlir::Value result;
    QualType lhsTy = e->getLHS()->getType();
    QualType rhsTy = e->getRHS()->getType();

    auto clangCmpToCirCmp = [](auto clangCmp) -> mlir::cir::CmpOpKind {
      switch (clangCmp) {
      case BO_LT:
        return mlir::cir::CmpOpKind::lt;
      case BO_GT:
        return mlir::cir::CmpOpKind::gt;
      case BO_LE:
        return mlir::cir::CmpOpKind::le;
      case BO_GE:
        return mlir::cir::CmpOpKind::ge;
      case BO_EQ:
        return mlir::cir::CmpOpKind::eq;
      case BO_NE:
        return mlir::cir::CmpOpKind::ne;
      default:
        llvm_unreachable("unsupported comparison kind");
        return mlir::cir::CmpOpKind(-1);
      }
    };

    if (const MemberPointerType *mpt = lhsTy->getAs<MemberPointerType>()) {
      assert(0 && "not implemented");
    } else if (!lhsTy->isAnyComplexType() && !rhsTy->isAnyComplexType()) {
      BinOpInfo boInfo = buildBinOps(e);
      mlir::Value lhs = boInfo.lhs;
      mlir::Value rhs = boInfo.rhs;

      if (lhsTy->isVectorType()) {
        if (!e->getType()->isVectorType()) {
          // If AltiVec, the comparison results in a numeric type, so we use
          // intrinsics comparing vectors and giving 0 or 1 as a result
          llvm_unreachable("NYI: AltiVec comparison");
        } else {
          // Other kinds of vectors.  Element-wise comparison returning
          // a vector.
          mlir::cir::CmpOpKind kind = clangCmpToCirCmp(e->getOpcode());
          return Builder.create<mlir::cir::VecCmpOp>(
              CGF.getLoc(boInfo.loc), CGF.getCIRType(boInfo.fullType), kind,
              boInfo.lhs, boInfo.rhs);
        }
      }
      if (boInfo.isFixedPointOp()) {
        assert(0 && "not implemented");
      } else {
        // FIXME(cir): handle another if above for CIR equivalent on
        // LHSTy->hasSignedIntegerRepresentation()

        // Unsigned integers and pointers.
        if (CGF.cgm.getCodeGenOpts().StrictVTablePointers &&
            mlir::isa<mlir::cir::PointerType>(lhs.getType()) &&
            mlir::isa<mlir::cir::PointerType>(rhs.getType())) {
          llvm_unreachable("NYI");
        }

        mlir::cir::CmpOpKind kind = clangCmpToCirCmp(e->getOpcode());
        return Builder.create<mlir::cir::CmpOp>(CGF.getLoc(boInfo.loc),
                                                CGF.getCIRType(boInfo.fullType),
                                                kind, boInfo.lhs, boInfo.rhs);
      }
    } else { // Complex Comparison: can only be an equality comparison.
      assert(0 && "not implemented");
    }

    return buildScalarConversion(result, CGF.getContext().BoolTy, e->getType(),
                                 e->getExprLoc());
  }

  mlir::Value buildFloatToBoolConversion(mlir::Value src, mlir::Location loc) {
    auto boolTy = Builder.getBoolTy();
    return Builder.create<mlir::cir::CastOp>(
        loc, boolTy, mlir::cir::CastKind::float_to_bool, src);
  }

  mlir::Value buildIntToBoolConversion(mlir::Value srcVal, mlir::Location loc) {
    // Because of the type rules of C, we often end up computing a
    // logical value, then zero extending it to int, then wanting it
    // as a logical value again.
    // TODO: optimize this common case here or leave it for later
    // CIR passes?
    mlir::Type boolTy = CGF.getCIRType(CGF.getContext().BoolTy);
    return Builder.create<mlir::cir::CastOp>(
        loc, boolTy, mlir::cir::CastKind::int_to_bool, srcVal);
  }

  /// Convert the specified expression value to a boolean (!cir.bool) truth
  /// value. This is equivalent to "Val != 0".
  mlir::Value buildConversionToBool(mlir::Value src, QualType srcType,
                                    mlir::Location loc) {
    assert(srcType.isCanonical() && "EmitScalarConversion strips typedefs");

    if (srcType->isRealFloatingType())
      return buildFloatToBoolConversion(src, loc);

    if (auto *mpt = llvm::dyn_cast<MemberPointerType>(srcType))
      assert(0 && "not implemented");

    if (srcType->isIntegerType())
      return buildIntToBoolConversion(src, loc);

    assert(::mlir::isa<::mlir::cir::PointerType>(src.getType()));
    return buildPointerToBoolConversion(src, srcType);
  }

  /// Emit a conversion from the specified type to the specified destination
  /// type, both of which are CIR scalar types.
  /// TODO: do we need ScalarConversionOpts here? Should be done in another
  /// pass.
  mlir::Value
  buildScalarConversion(mlir::Value src, QualType srcType, QualType dstType,
                        SourceLocation loc,
                        ScalarConversionOpts opts = ScalarConversionOpts()) {
    // All conversions involving fixed point types should be handled by the
    // buildFixedPoint family functions. This is done to prevent bloating up
    // this function more, and although fixed point numbers are represented by
    // integers, we do not want to follow any logic that assumes they should be
    // treated as integers.
    // TODO(leonardchan): When necessary, add another if statement checking for
    // conversions to fixed point types from other types.
    if (srcType->isFixedPointType()) {
      llvm_unreachable("not implemented");
    } else if (dstType->isFixedPointType()) {
      llvm_unreachable("not implemented");
    }

    srcType = CGF.getContext().getCanonicalType(srcType);
    dstType = CGF.getContext().getCanonicalType(dstType);
    if (srcType == dstType)
      return src;

    if (dstType->isVoidType())
      return nullptr;

    mlir::Type srcTy = src.getType();

    // Handle conversions to bool first, they are special: comparisons against
    // 0.
    if (dstType->isBooleanType())
      return buildConversionToBool(src, srcType, CGF.getLoc(loc));

    mlir::Type dstTy = convertType(dstType);

    // Cast from half through float if half isn't a native type.
    if (srcType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      llvm_unreachable("not implemented");
    }

    // TODO(cir): LLVM codegen ignore conversions like int -> uint,
    // is there anything to be done for CIR here?
    if (srcTy == dstTy) {
      if (opts.emitImplicitIntegerSignChangeChecks)
        llvm_unreachable("not implemented");
      return src;
    }

    // Handle pointer conversions next: pointers can only be converted to/from
    // other pointers and integers. Check for pointer types in terms of LLVM, as
    // some native types (like Obj-C id) may map to a pointer type.
    if (auto dstPt = dyn_cast<mlir::cir::PointerType>(dstTy)) {
      llvm_unreachable("NYI");
    }

    if (isa<mlir::cir::PointerType>(srcTy)) {
      // Must be an ptr to int cast.
      assert(isa<mlir::cir::IntType>(dstTy) && "not ptr->int?");
      return Builder.createPtrToInt(src, dstTy);
    }

    // A scalar can be splatted to an extended vector of the same element type
    if (dstType->isExtVectorType() && !srcType->isVectorType()) {
      // Sema should add casts to make sure that the source expression's type
      // is the same as the vector's element type (sans qualifiers)
      assert(dstType->castAs<ExtVectorType>()->getElementType().getTypePtr() ==
                 srcType.getTypePtr() &&
             "Splatted expr doesn't match with vector element type?");

      llvm_unreachable("not implemented");
    }

    if (srcType->isMatrixType() && dstType->isMatrixType())
      llvm_unreachable("NYI: matrix type to matrix type conversion");
    assert(!srcType->isMatrixType() && !dstType->isMatrixType() &&
           "Internal error: conversion between matrix type and scalar type");

    // Finally, we have the arithmetic types or vectors of arithmetic types.
    mlir::Value res = nullptr;
    mlir::Type resTy = dstTy;

    // An overflowing conversion has undefined behavior if eitehr the source
    // type or the destination type is a floating-point type. However, we
    // consider the range of representable values for all floating-point types
    // to be [-inf,+inf], so no overflow can ever happen when the destination
    // type is a floating-point type.
    if (CGF.SanOpts.has(SanitizerKind::FloatCastOverflow))
      llvm_unreachable("NYI");

    // Cast to half through float if half isn't a native type.
    if (dstType->isHalfType() &&
        !CGF.getContext().getLangOpts().NativeHalfType) {
      llvm_unreachable("NYI");
    }

    res = buildScalarCast(src, srcType, dstType, srcTy, dstTy, opts);

    if (dstTy != resTy) {
      llvm_unreachable("NYI");
    }

    if (opts.emitImplicitIntegerTruncationChecks)
      llvm_unreachable("NYI");

    if (opts.emitImplicitIntegerSignChangeChecks)
      llvm_unreachable("NYI");

    return res;
  }
};

} // namespace

/// Emit the computation of the specified expression of scalar type,
/// ignoring the result.
mlir::Value CIRGenFunction::buildScalarExpr(const Expr *e) {
  assert(e && hasScalarEvaluationKind(e->getType()) &&
         "Invalid scalar expression to emit");

  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(e));
}

mlir::Value CIRGenFunction::buildPromotedScalarExpr(const Expr *e,
                                                    QualType promotionType) {
  if (!promotionType.isNull())
    return ScalarExprEmitter(*this, builder).buildPromoted(e, promotionType);
  return ScalarExprEmitter(*this, builder).Visit(const_cast<Expr *>(e));
}

[[maybe_unused]] static bool mustVisitNullValue(const Expr *e) {
  // If a null pointer expression's type is the C++0x nullptr_t, then
  // it's not necessarily a simple constant and it must be evaluated
  // for its potential side effects.
  return e->getType()->isNullPtrType();
}

/// If \p E is a widened promoted integer, get its base (unpromoted) type.
static std::optional<QualType> getUnwidenedIntegerType(const ASTContext &ctx,
                                                       const Expr *e) {
  const Expr *base = e->IgnoreImpCasts();
  if (e == base)
    return std::nullopt;

  QualType baseTy = base->getType();
  if (!ctx.isPromotableIntegerType(baseTy) ||
      ctx.getTypeSize(baseTy) >= ctx.getTypeSize(e->getType()))
    return std::nullopt;

  return baseTy;
}

/// Check if \p E is a widened promoted integer.
[[maybe_unused]] static bool isWidenedIntegerOp(const ASTContext &ctx,
                                                const Expr *e) {
  return getUnwidenedIntegerType(ctx, e).has_value();
}

/// Check if we can skip the overflow check for \p Op.
[[maybe_unused]] static bool canElideOverflowCheck(const ASTContext &ctx,
                                                   const BinOpInfo &op) {
  assert((isa<UnaryOperator>(op.e) || isa<BinaryOperator>(op.e)) &&
         "Expected a unary or binary operator");

  // If the binop has constant inputs and we can prove there is no overflow,
  // we can elide the overflow check.
  if (!op.mayHaveIntegerOverflow())
    return true;

  // If a unary op has a widened operand, the op cannot overflow.
  if (const auto *uo = dyn_cast<UnaryOperator>(op.e))
    return !uo->canOverflow();

  // We usually don't need overflow checks for binops with widened operands.
  // Multiplication with promoted unsigned operands is a special case.
  const auto *bo = cast<BinaryOperator>(op.e);
  auto optionalLhsTy = getUnwidenedIntegerType(ctx, bo->getLHS());
  if (!optionalLhsTy)
    return false;

  auto optionalRhsTy = getUnwidenedIntegerType(ctx, bo->getRHS());
  if (!optionalRhsTy)
    return false;

  QualType lhsTy = *optionalLhsTy;
  QualType rhsTy = *optionalRhsTy;

  // This is the simple case: binops without unsigned multiplication, and with
  // widened operands. No overflow check is needed here.
  if ((op.opcode != BO_Mul && op.opcode != BO_MulAssign) ||
      !lhsTy->isUnsignedIntegerType() || !rhsTy->isUnsignedIntegerType())
    return true;

  // For unsigned multiplication the overflow check can be elided if either one
  // of the unpromoted types are less than half the size of the promoted type.
  unsigned promotedSize = ctx.getTypeSize(op.e->getType());
  return (2 * ctx.getTypeSize(lhsTy)) < promotedSize ||
         (2 * ctx.getTypeSize(rhsTy)) < promotedSize;
}

/// Emit pointer + index arithmetic.
static mlir::Value buildPointerArithmetic(CIRGenFunction &cgf,
                                          const BinOpInfo &op,
                                          bool isSubtraction) {
  // Must have binary (not unary) expr here.  Unary pointer
  // increment/decrement doesn't use this path.
  const BinaryOperator *expr = cast<BinaryOperator>(op.e);

  mlir::Value pointer = op.lhs;
  Expr *pointerOperand = expr->getLHS();
  mlir::Value index = op.rhs;
  Expr *indexOperand = expr->getRHS();

  // In a subtraction, the LHS is always the pointer.
  if (!isSubtraction && !mlir::isa<mlir::cir::PointerType>(pointer.getType())) {
    std::swap(pointer, index);
    std::swap(pointerOperand, indexOperand);
  }

  bool isSigned = indexOperand->getType()->isSignedIntegerOrEnumerationType();

  // Some versions of glibc and gcc use idioms (particularly in their malloc
  // routines) that add a pointer-sized integer (known to be a pointer value)
  // to a null pointer in order to cast the value back to an integer or as
  // part of a pointer alignment algorithm.  This is undefined behavior, but
  // we'd like to be able to compile programs that use it.
  //
  // Normally, we'd generate a GEP with a null-pointer base here in response
  // to that code, but it's also UB to dereference a pointer created that
  // way.  Instead (as an acknowledged hack to tolerate the idiom) we will
  // generate a direct cast of the integer value to a pointer.
  //
  // The idiom (p = nullptr + N) is not met if any of the following are true:
  //
  //   The operation is subtraction.
  //   The index is not pointer-sized.
  //   The pointer type is not byte-sized.
  //
  if (BinaryOperator::isNullPointerArithmeticExtension(
          cgf.getContext(), op.opcode, expr->getLHS(), expr->getRHS()))
    return cgf.getBuilder().createIntToPtr(index, pointer.getType());

  // Differently from LLVM codegen, ABI bits for index sizes is handled during
  // LLVM lowering.

  // If this is subtraction, negate the index.
  if (isSubtraction)
    index = cgf.getBuilder().createNeg(index);

  if (cgf.SanOpts.has(SanitizerKind::ArrayBounds))
    llvm_unreachable("array bounds sanitizer is NYI");

  const PointerType *pointerType =
      pointerOperand->getType()->getAs<PointerType>();
  if (!pointerType)
    llvm_unreachable("ObjC is NYI");

  QualType elementType = pointerType->getPointeeType();
  if (const VariableArrayType *vla =
          cgf.getContext().getAsVariableArrayType(elementType)) {

    // The element count here is the total number of non-VLA elements.
    mlir::Value numElements = cgf.getVLASize(vla).NumElts;

    // GEP indexes are signed, and scaling an index isn't permitted to
    // signed-overflow, so we use the same semantics for our explicit
    // multiply.  We suppress this if overflow is not undefined behavior.
    mlir::Type elemTy = cgf.convertTypeForMem(vla->getElementType());

    index = cgf.getBuilder().createCast(mlir::cir::CastKind::integral, index,
                                        numElements.getType());
    index = cgf.getBuilder().createMul(index, numElements);

    if (cgf.getLangOpts().isSignedOverflowDefined()) {
      pointer = cgf.getBuilder().create<mlir::cir::PtrStrideOp>(
          cgf.getLoc(op.e->getExprLoc()), pointer.getType(), pointer, index);
    } else {
      pointer = cgf.buildCheckedInBoundsGEP(elemTy, pointer, index, isSigned,
                                            isSubtraction, op.e->getExprLoc());
    }
    return pointer;
  }
  // Explicitly handle GNU void* and function pointer arithmetic extensions. The
  // GNU void* casts amount to no-ops since our void* type is i8*, but this is
  // future proof.
  mlir::Type elemTy;
  if (elementType->isVoidType() || elementType->isFunctionType())
    elemTy = cgf.UInt8Ty;
  else
    elemTy = cgf.convertTypeForMem(elementType);

  if (cgf.getLangOpts().isSignedOverflowDefined())
    return cgf.getBuilder().create<mlir::cir::PtrStrideOp>(
        cgf.getLoc(op.e->getExprLoc()), pointer.getType(), pointer, index);

  return cgf.buildCheckedInBoundsGEP(elemTy, pointer, index, isSigned,
                                     isSubtraction, op.e->getExprLoc());
}

mlir::Value ScalarExprEmitter::buildMul(const BinOpInfo &ops) {
  if (ops.compType->isSignedIntegerOrEnumerationType()) {
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createMul(ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createNSWMul(ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      if (canElideOverflowCheck(CGF.getContext(), ops))
        return Builder.createNSWMul(ops.lhs, ops.rhs);
      llvm_unreachable("NYI");
    }
  }
  if (ops.fullType->isConstantMatrixType()) {
    llvm_unreachable("NYI");
  }
  if (ops.compType->isUnsignedIntegerType() &&
      CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
      !canElideOverflowCheck(CGF.getContext(), ops))
    llvm_unreachable("NYI");

  if (mlir::cir::isFPOrFPVectorTy(ops.lhs.getType())) {
    CIRGenFunction::CIRGenFPOptionsRAII fpOptsRaii(CGF, ops.fpFeatures);
    return Builder.createFMul(ops.lhs, ops.rhs);
  }

  if (ops.isFixedPointOp())
    llvm_unreachable("NYI");

  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
      mlir::cir::BinOpKind::Mul, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::buildDiv(const BinOpInfo &ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
      mlir::cir::BinOpKind::Div, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::buildRem(const BinOpInfo &ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
      mlir::cir::BinOpKind::Rem, ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::buildAdd(const BinOpInfo &ops) {
  if (mlir::isa<mlir::cir::PointerType>(ops.lhs.getType()) ||
      mlir::isa<mlir::cir::PointerType>(ops.rhs.getType()))
    return buildPointerArithmetic(CGF, ops, /*isSubtraction=*/false);
  if (ops.compType->isSignedIntegerOrEnumerationType()) {
    switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
    case LangOptions::SOB_Defined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createAdd(ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Undefined:
      if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
        return Builder.createNSWAdd(ops.lhs, ops.rhs);
      [[fallthrough]];
    case LangOptions::SOB_Trapping:
      if (canElideOverflowCheck(CGF.getContext(), ops))
        return Builder.createNSWAdd(ops.lhs, ops.rhs);

      llvm_unreachable("NYI");
    }
  }
  if (ops.fullType->isConstantMatrixType()) {
    llvm_unreachable("NYI");
  }

  if (ops.compType->isUnsignedIntegerType() &&
      CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
      !canElideOverflowCheck(CGF.getContext(), ops))
    llvm_unreachable("NYI");

  if (mlir::cir::isFPOrFPVectorTy(ops.lhs.getType())) {
    CIRGenFunction::CIRGenFPOptionsRAII fpOptsRaii(CGF, ops.fpFeatures);
    return Builder.createFAdd(ops.lhs, ops.rhs);
  }

  if (ops.isFixedPointOp())
    llvm_unreachable("NYI");

  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
      mlir::cir::BinOpKind::Add, ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::buildSub(const BinOpInfo &ops) {
  // The LHS is always a pointer if either side is.
  if (!mlir::isa<mlir::cir::PointerType>(ops.lhs.getType())) {
    if (ops.compType->isSignedIntegerOrEnumerationType()) {
      switch (CGF.getLangOpts().getSignedOverflowBehavior()) {
      case LangOptions::SOB_Defined: {
        if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return Builder.createSub(ops.lhs, ops.rhs);
        [[fallthrough]];
      }
      case LangOptions::SOB_Undefined:
        if (!CGF.SanOpts.has(SanitizerKind::SignedIntegerOverflow))
          return Builder.createNSWSub(ops.lhs, ops.rhs);
        [[fallthrough]];
      case LangOptions::SOB_Trapping:
        if (canElideOverflowCheck(CGF.getContext(), ops))
          return Builder.createNSWSub(ops.lhs, ops.rhs);
        llvm_unreachable("NYI");
      }
    }

    if (ops.fullType->isConstantMatrixType()) {
      llvm_unreachable("NYI");
    }

    if (ops.compType->isUnsignedIntegerType() &&
        CGF.SanOpts.has(SanitizerKind::UnsignedIntegerOverflow) &&
        !canElideOverflowCheck(CGF.getContext(), ops))
      llvm_unreachable("NYI");

    if (mlir::cir::isFPOrFPVectorTy(ops.lhs.getType())) {
      CIRGenFunction::CIRGenFPOptionsRAII fpOptsRaii(CGF, ops.fpFeatures);
      return Builder.createFSub(ops.lhs, ops.rhs);
    }

    if (ops.isFixedPointOp())
      llvm_unreachable("NYI");

    return Builder.create<mlir::cir::BinOp>(
        CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
        mlir::cir::BinOpKind::Sub, ops.lhs, ops.rhs);
  }

  // If the RHS is not a pointer, then we have normal pointer
  // arithmetic.
  if (!mlir::isa<mlir::cir::PointerType>(ops.rhs.getType()))
    return buildPointerArithmetic(CGF, ops, /*isSubtraction=*/true);

  // Otherwise, this is a pointer subtraction

  // Do the raw subtraction part.
  //
  // TODO(cir): note for LLVM lowering out of this; when expanding this into
  // LLVM we shall take VLA's, division by element size, etc.
  //
  // See more in `EmitSub` in CGExprScalar.cpp.
  assert(!MissingFeatures::llvmLoweringPtrDiffConsidersPointee());
  return Builder.create<mlir::cir::PtrDiffOp>(CGF.getLoc(ops.loc),
                                              CGF.PtrDiffTy, ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::buildShl(const BinOpInfo &ops) {
  // TODO: This misses out on the sanitizer check below.
  if (ops.isFixedPointOp())
    llvm_unreachable("NYI");

  // CIR accepts shift between different types, meaning nothing special
  // to be done here. OTOH, LLVM requires the LHS and RHS to be the same type:
  // promote or truncate the RHS to the same size as the LHS.

  bool sanitizeSignedBase = CGF.SanOpts.has(SanitizerKind::ShiftBase) &&
                            ops.compType->hasSignedIntegerRepresentation() &&
                            !CGF.getLangOpts().isSignedOverflowDefined() &&
                            !CGF.getLangOpts().CPlusPlus20;
  bool sanitizeUnsignedBase =
      CGF.SanOpts.has(SanitizerKind::UnsignedShiftBase) &&
      ops.compType->hasUnsignedIntegerRepresentation();
  bool sanitizeBase = sanitizeSignedBase || sanitizeUnsignedBase;
  bool sanitizeExponent = CGF.SanOpts.has(SanitizerKind::ShiftExponent);

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (CGF.getLangOpts().OpenCL)
    llvm_unreachable("NYI");
  else if ((sanitizeBase || sanitizeExponent) &&
           mlir::isa<mlir::cir::IntType>(ops.lhs.getType())) {
    llvm_unreachable("NYI");
  }

  return Builder.create<mlir::cir::ShiftOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType), ops.lhs, ops.rhs,
      CGF.getBuilder().getUnitAttr());
}

mlir::Value ScalarExprEmitter::buildShr(const BinOpInfo &ops) {
  // TODO: This misses out on the sanitizer check below.
  if (ops.isFixedPointOp())
    llvm_unreachable("NYI");

  // CIR accepts shift between different types, meaning nothing special
  // to be done here. OTOH, LLVM requires the LHS and RHS to be the same type:
  // promote or truncate the RHS to the same size as the LHS.

  // OpenCL 6.3j: shift values are effectively % word size of LHS.
  if (CGF.getLangOpts().OpenCL)
    llvm_unreachable("NYI");
  else if (CGF.SanOpts.has(SanitizerKind::ShiftExponent) &&
           mlir::isa<mlir::cir::IntType>(ops.lhs.getType())) {
    llvm_unreachable("NYI");
  }

  // Note that we don't need to distinguish unsigned treatment at this
  // point since it will be handled later by LLVM lowering.
  return Builder.create<mlir::cir::ShiftOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType), ops.lhs, ops.rhs);
}

mlir::Value ScalarExprEmitter::buildAnd(const BinOpInfo &ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
      mlir::cir::BinOpKind::And, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::buildXor(const BinOpInfo &ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
      mlir::cir::BinOpKind::Xor, ops.lhs, ops.rhs);
}
mlir::Value ScalarExprEmitter::buildOr(const BinOpInfo &ops) {
  return Builder.create<mlir::cir::BinOp>(
      CGF.getLoc(ops.loc), CGF.getCIRType(ops.fullType),
      mlir::cir::BinOpKind::Or, ops.lhs, ops.rhs);
}

// Emit code for an explicit or implicit cast.  Implicit
// casts have to handle a more broad range of conversions than explicit
// casts, as they handle things like function to ptr-to-function decay
// etc.
mlir::Value ScalarExprEmitter::VisitCastExpr(CastExpr *ce) {
  Expr *e = ce->getSubExpr();
  QualType destTy = ce->getType();
  CastKind kind = ce->getCastKind();

  // These cases are generally not written to ignore the result of evaluating
  // their sub-expressions, so we clear this now.
  bool ignored = testAndClearIgnoreResultAssign();
  (void)ignored;

  // Since almost all cast kinds apply to scalars, this switch doesn't have a
  // default case, so the compiler will warn on a missing case. The cases are
  // in the same order as in the CastKind enum.
  switch (kind) {
  case clang::CK_Dependent:
    llvm_unreachable("dependent cast kind in CIR gen!");
  case clang::CK_BuiltinFnToFnPtr:
    llvm_unreachable("builtin functions are handled elsewhere");

  case CK_LValueBitCast:
  case CK_ObjCObjectLValueCast:
  case CK_LValueToRValueBitCast: {
    LValue sourceLVal = CGF.buildLValue(e);
    Address sourceAddr = sourceLVal.getAddress();

    mlir::Type destElemTy = CGF.convertTypeForMem(destTy);
    mlir::Type destPtrTy = CGF.getBuilder().getPointerTo(destElemTy);
    mlir::Value destPtr = CGF.getBuilder().createBitcast(
        CGF.getLoc(e->getExprLoc()), sourceAddr.getPointer(), destPtrTy);

    Address destAddr =
        sourceAddr.withPointer(destPtr).withElementType(destElemTy);
    LValue destLVal = CGF.makeAddrLValue(destAddr, destTy);

    if (kind == CK_LValueToRValueBitCast)
      assert(!MissingFeatures::tbaa());

    return buildLoadOfLValue(destLVal, ce->getExprLoc());
  }

  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_BitCast: {
    auto src = Visit(const_cast<Expr *>(e));
    mlir::Type dstTy = CGF.convertType(destTy);

    assert(!MissingFeatures::addressSpace());
    if (CGF.SanOpts.has(SanitizerKind::CFIUnrelatedCast)) {
      llvm_unreachable("NYI");
    }

    if (CGF.cgm.getCodeGenOpts().StrictVTablePointers) {
      llvm_unreachable("NYI");
    }

    // Update heapallocsite metadata when there is an explicit pointer cast.
    assert(!MissingFeatures::addHeapAllocSiteMetadata());

    // If Src is a fixed vector and Dst is a scalable vector, and both have the
    // same element type, use the llvm.vector.insert intrinsic to perform the
    // bitcast.
    assert(!MissingFeatures::scalableVectors());

    // If Src is a scalable vector and Dst is a fixed vector, and both have the
    // same element type, use the llvm.vector.extract intrinsic to perform the
    // bitcast.
    assert(!MissingFeatures::scalableVectors());

    // Perform VLAT <-> VLST bitcast through memory.
    // TODO: since the llvm.experimental.vector.{insert,extract} intrinsics
    //       require the element types of the vectors to be the same, we
    //       need to keep this around for bitcasts between VLAT <-> VLST where
    //       the element types of the vectors are not the same, until we figure
    //       out a better way of doing these casts.
    assert(!MissingFeatures::scalableVectors());

    return CGF.getBuilder().createBitcast(CGF.getLoc(e->getSourceRange()), src,
                                          dstTy);
  }
  case CK_AddressSpaceConversion: {
    Expr::EvalResult result;
    if (e->EvaluateAsRValue(result, CGF.getContext()) &&
        result.Val.isNullPointer()) {
      // If E has side effect, it is emitted even if its final result is a
      // null pointer. In that case, a DCE pass should be able to
      // eliminate the useless instructions emitted during translating E.
      if (result.HasSideEffects)
        Visit(e);
      return CGF.cgm.buildNullConstant(destTy, CGF.getLoc(e->getExprLoc()));
    }
    // Since target may map different address spaces in AST to the same address
    // space, an address space conversion may end up as a bitcast.
    auto srcAs = CGF.builder.getAddrSpaceAttr(
        e->getType()->getPointeeType().getAddressSpace());
    auto destAs = CGF.builder.getAddrSpaceAttr(
        destTy->getPointeeType().getAddressSpace());
    return CGF.cgm.getTargetCIRGenInfo().performAddrSpaceCast(
        CGF, Visit(e), srcAs, destAs, convertType(destTy));
  }
  case CK_AtomicToNonAtomic:
    llvm_unreachable("NYI");
  case CK_NonAtomicToAtomic:
  case CK_UserDefinedConversion:
    return Visit(const_cast<Expr *>(e));
  case CK_NoOp: {
    auto v = Visit(const_cast<Expr *>(e));
    if (v) {
      // CK_NoOp can model a pointer qualification conversion, which can remove
      // an array bound and change the IR type.
      // FIXME: Once pointee types are removed from IR, remove this.
      auto t = CGF.convertType(destTy);
      if (t != v.getType())
        assert(0 && "NYI");
    }
    return v;
  }
  case CK_BaseToDerived:
    llvm_unreachable("NYI");
  case CK_DerivedToBase: {
    // The EmitPointerWithAlignment path does this fine; just discard
    // the alignment.
    return CGF.buildPointerWithAlignment(ce).getPointer();
  }
  case CK_Dynamic: {
    Address v = CGF.buildPointerWithAlignment(e);
    const auto *dce = cast<CXXDynamicCastExpr>(ce);
    return CGF.buildDynamicCast(v, dce);
  }
  case CK_ArrayToPointerDecay:
    return CGF.buildArrayToPointerDecay(e).getPointer();
  case CK_FunctionToPointerDecay:
    return buildLValue(e).getPointer();

  case CK_NullToPointer: {
    // FIXME: use MustVisitNullValue(E) and evaluate expr.
    // Note that DestTy is used as the MLIR type instead of a custom
    // nullptr type.
    mlir::Type ty = CGF.getCIRType(destTy);
    return Builder.getNullPtr(ty, CGF.getLoc(e->getExprLoc()));
  }

  case CK_NullToMemberPointer: {
    if (mustVisitNullValue(e))
      CGF.buildIgnoredExpr(e);

    assert(!MissingFeatures::cxxABI());

    const MemberPointerType *mpt = ce->getType()->getAs<MemberPointerType>();
    if (mpt->isMemberFunctionPointerType()) {
      auto ty = mlir::cast<mlir::cir::MethodType>(CGF.getCIRType(destTy));
      return Builder.getNullMethodPtr(ty, CGF.getLoc(e->getExprLoc()));
    }

    auto ty = mlir::cast<mlir::cir::DataMemberType>(CGF.getCIRType(destTy));
    return Builder.getNullDataMemberPtr(ty, CGF.getLoc(e->getExprLoc()));
  }
  case CK_ReinterpretMemberPointer:
    llvm_unreachable("NYI");
  case CK_BaseToDerivedMemberPointer:
    llvm_unreachable("NYI");
  case CK_DerivedToBaseMemberPointer:
    llvm_unreachable("NYI");
  case CK_ARCProduceObject:
    llvm_unreachable("NYI");
  case CK_ARCConsumeObject:
    llvm_unreachable("NYI");
  case CK_ARCReclaimReturnedObject:
    llvm_unreachable("NYI");
  case CK_ARCExtendBlockObject:
    llvm_unreachable("NYI");
  case CK_CopyAndAutoreleaseBlockObject:
    llvm_unreachable("NYI");

  case CK_FloatingRealToComplex:
  case CK_FloatingComplexCast:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_FloatingComplexToIntegralComplex:
    llvm_unreachable("scalar cast to non-scalar value");

  case CK_ConstructorConversion:
    llvm_unreachable("NYI");
  case CK_ToUnion:
    llvm_unreachable("NYI");

  case CK_LValueToRValue:
    assert(CGF.getContext().hasSameUnqualifiedType(e->getType(), destTy));
    assert(e->isGLValue() && "lvalue-to-rvalue applied to r-value!");
    return Visit(const_cast<Expr *>(e));

  case CK_IntegralToPointer: {
    auto destCirTy = convertType(destTy);
    mlir::Value src = Visit(const_cast<Expr *>(e));

    // Properly resize by casting to an int of the same size as the pointer.
    // Clang's IntegralToPointer includes 'bool' as the source, but in CIR
    // 'bool' is not an integral type.  So check the source type to get the
    // correct CIR conversion.
    auto middleTy = CGF.cgm.getDataLayout().getIntPtrType(destCirTy);
    auto middleVal = Builder.createCast(e->getType()->isBooleanType()
                                            ? mlir::cir::CastKind::bool_to_int
                                            : mlir::cir::CastKind::integral,
                                        src, middleTy);

    if (CGF.cgm.getCodeGenOpts().StrictVTablePointers)
      llvm_unreachable("NYI");

    return Builder.createIntToPtr(middleVal, destCirTy);
  }
  case CK_PointerToIntegral: {
    assert(!destTy->isBooleanType() && "bool should use PointerToBool");
    if (CGF.cgm.getCodeGenOpts().StrictVTablePointers)
      llvm_unreachable("NYI");
    return Builder.createPtrToInt(Visit(e), convertType(destTy));
  }
  case CK_ToVoid: {
    CGF.buildIgnoredExpr(e);
    return nullptr;
  }
  case CK_MatrixCast:
    llvm_unreachable("NYI");
  case CK_VectorSplat: {
    // Create a vector object and fill all elements with the same scalar value.
    assert(destTy->isVectorType() && "CK_VectorSplat to non-vector type");
    return CGF.getBuilder().create<mlir::cir::VecSplatOp>(
        CGF.getLoc(e->getSourceRange()), CGF.getCIRType(destTy), Visit(e));
  }
  case CK_FixedPointCast:
    llvm_unreachable("NYI");
  case CK_FixedPointToBoolean:
    llvm_unreachable("NYI");
  case CK_FixedPointToIntegral:
    llvm_unreachable("NYI");
  case CK_IntegralToFixedPoint:
    llvm_unreachable("NYI");

  case CK_IntegralCast: {
    ScalarConversionOpts opts;
    if (auto *ice = dyn_cast<ImplicitCastExpr>(ce)) {
      if (!ice->isPartOfExplicitCast())
        opts = ScalarConversionOpts(CGF.SanOpts);
    }
    return buildScalarConversion(Visit(e), e->getType(), destTy,
                                 ce->getExprLoc(), opts);
  }

  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingCast:
  case CK_FixedPointToFloating:
  case CK_FloatingToFixedPoint: {
    if (kind == CK_FixedPointToFloating || kind == CK_FloatingToFixedPoint)
      llvm_unreachable("Fixed point casts are NYI.");
    CIRGenFunction::CIRGenFPOptionsRAII fpOptsRaii(CGF, ce);
    return buildScalarConversion(Visit(e), e->getType(), destTy,
                                 ce->getExprLoc());
  }
  case CK_BooleanToSignedIntegral:
    llvm_unreachable("NYI");

  case CK_IntegralToBoolean: {
    return buildIntToBoolConversion(Visit(e), CGF.getLoc(ce->getSourceRange()));
  }

  case CK_PointerToBoolean:
    return buildPointerToBoolConversion(Visit(e), e->getType());
  case CK_FloatingToBoolean:
    return buildFloatToBoolConversion(Visit(e), CGF.getLoc(e->getExprLoc()));
  case CK_MemberPointerToBoolean:
    llvm_unreachable("NYI");
  case CK_FloatingComplexToReal:
  case CK_IntegralComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToBoolean: {
    mlir::Value v = CGF.buildComplexExpr(e);
    return buildComplexToScalarConversion(CGF.getLoc(ce->getExprLoc()), v, kind,
                                          destTy);
  }
  case CK_ZeroToOCLOpaqueType:
    llvm_unreachable("NYI");
  case CK_IntToOCLSampler:
    llvm_unreachable("NYI");

  default:
    emitError(CGF.getLoc(ce->getExprLoc()), "cast kind not implemented: '")
        << ce->getCastKindName() << "'";
    return nullptr;
  } // end of switch

  llvm_unreachable("unknown scalar cast");
}

mlir::Value ScalarExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(CGF.getContext())->isReferenceType())
    return buildLoadOfLValue(e);

  auto v = CGF.buildCallExpr(e).getScalarVal();
  assert(!MissingFeatures::buildLValueAlignmentAssumption());
  return v;
}

mlir::Value ScalarExprEmitter::VisitMemberExpr(MemberExpr *e) {
  // TODO(cir): Folding all this constants sound like work for MLIR optimizers,
  // keep assertion for now.
  assert(!MissingFeatures::tryEmitAsConstant());
  Expr::EvalResult result;
  if (e->EvaluateAsInt(result, CGF.getContext(), Expr::SE_AllowSideEffects)) {
    llvm::APSInt value = result.Val.getInt();
    CGF.buildIgnoredExpr(e->getBase());
    return Builder.getConstInt(CGF.getLoc(e->getExprLoc()), value);
  }
  return buildLoadOfLValue(e);
}

/// Emit a conversion from the specified type to the specified destination
/// type, both of which are CIR scalar types.
mlir::Value CIRGenFunction::buildScalarConversion(mlir::Value src,
                                                  QualType srcTy,
                                                  QualType dstTy,
                                                  SourceLocation loc) {
  assert(CIRGenFunction::hasScalarEvaluationKind(srcTy) &&
         CIRGenFunction::hasScalarEvaluationKind(dstTy) &&
         "Invalid scalar expression to emit");
  return ScalarExprEmitter(*this, builder)
      .buildScalarConversion(src, srcTy, dstTy, loc);
}

mlir::Value CIRGenFunction::buildComplexToScalarConversion(mlir::Value src,
                                                           QualType srcTy,
                                                           QualType dstTy,
                                                           SourceLocation loc) {
  assert(srcTy->isAnyComplexType() && hasScalarEvaluationKind(dstTy) &&
         "Invalid complex -> scalar conversion");

  auto complexElemTy = srcTy->castAs<ComplexType>()->getElementType();
  if (dstTy->isBooleanType()) {
    auto kind = complexElemTy->isFloatingType()
                    ? mlir::cir::CastKind::float_complex_to_bool
                    : mlir::cir::CastKind::int_complex_to_bool;
    return builder.createCast(getLoc(loc), kind, src, ConvertType(dstTy));
  }

  auto kind = complexElemTy->isFloatingType()
                  ? mlir::cir::CastKind::float_complex_to_real
                  : mlir::cir::CastKind::int_complex_to_real;
  auto real =
      builder.createCast(getLoc(loc), kind, src, ConvertType(complexElemTy));
  return buildScalarConversion(real, complexElemTy, dstTy, loc);
}

/// If the specified expression does not fold
/// to a constant, or if it does but contains a label, return false.  If it
/// constant folds return true and set the boolean result in Result.
bool CIRGenFunction::ConstantFoldsToSimpleInteger(const Expr *cond,
                                                  bool &resultBool,
                                                  bool allowLabels) {
  llvm::APSInt resultInt;
  if (!ConstantFoldsToSimpleInteger(cond, resultInt, allowLabels))
    return false;

  resultBool = resultInt.getBoolValue();
  return true;
}

mlir::Value ScalarExprEmitter::VisitInitListExpr(InitListExpr *e) {
  bool ignore = testAndClearIgnoreResultAssign();
  (void)ignore;
  assert(ignore == false && "init list ignored");
  unsigned numInitElements = e->getNumInits();

  if (e->hadArrayRangeDesignator())
    llvm_unreachable("NYI");

  if (e->getType()->isVectorType()) {
    assert(!MissingFeatures::scalableVectors() && "NYI: scalable vector init");
    assert(!MissingFeatures::vectorConstants() && "NYI: vector constants");
    auto vectorType =
        mlir::dyn_cast<mlir::cir::VectorType>(CGF.getCIRType(e->getType()));
    SmallVector<mlir::Value, 16> elements;
    for (Expr *init : e->inits()) {
      elements.push_back(Visit(init));
    }
    // Zero-initialize any remaining values.
    if (numInitElements < vectorType.getSize()) {
      mlir::Value zeroValue = CGF.getBuilder().create<mlir::cir::ConstantOp>(
          CGF.getLoc(e->getSourceRange()), vectorType.getEltType(),
          CGF.getBuilder().getZeroInitAttr(vectorType.getEltType()));
      for (uint64_t i = numInitElements; i < vectorType.getSize(); ++i) {
        elements.push_back(zeroValue);
      }
    }
    return CGF.getBuilder().create<mlir::cir::VecCreateOp>(
        CGF.getLoc(e->getSourceRange()), vectorType, elements);
  }

  if (numInitElements == 0) {
    // C++11 value-initialization for the scalar.
    return buildNullValue(e->getType(), CGF.getLoc(e->getExprLoc()));
  }

  return Visit(e->getInit(0));
}

mlir::Value ScalarExprEmitter::VisitUnaryLNot(const UnaryOperator *e) {
  // Perform vector logical not on comparison with zero vector.
  if (e->getType()->isVectorType() &&
      e->getType()->castAs<VectorType>()->getVectorKind() ==
          VectorKind::Generic) {
    llvm_unreachable("NYI");
  }

  // Compare operand to zero.
  mlir::Value boolVal = CGF.evaluateExprAsBool(e->getSubExpr());

  // Invert value.
  boolVal = Builder.createNot(boolVal);

  // ZExt result to the expr type.
  auto dstTy = convertType(e->getType());
  if (mlir::isa<mlir::cir::IntType>(dstTy))
    return Builder.createBoolToInt(boolVal, dstTy);
  if (mlir::isa<mlir::cir::BoolType>(dstTy))
    return boolVal;

  llvm_unreachable("destination type for logical-not unary operator is NYI");
}

mlir::Value ScalarExprEmitter::visitReal(const UnaryOperator *e) {
  // TODO(cir): handle scalar promotion.

  Expr *op = e->getSubExpr();
  if (op->getType()->isAnyComplexType()) {
    // If it's an l-value, load through the appropriate subobject l-value.
    // Note that we have to ask E because Op might be an l-value that
    // this won't work for, e.g. an Obj-C property.
    if (e->isGLValue())
      return CGF.buildLoadOfLValue(CGF.buildLValue(e), e->getExprLoc())
          .getScalarVal();
    // Otherwise, calculate and project.
    llvm_unreachable("NYI");
  }

  return Visit(op);
}

mlir::Value ScalarExprEmitter::visitImag(const UnaryOperator *e) {
  // TODO(cir): handle scalar promotion.

  Expr *op = e->getSubExpr();
  if (op->getType()->isAnyComplexType()) {
    // If it's an l-value, load through the appropriate subobject l-value.
    // Note that we have to ask E because Op might be an l-value that
    // this won't work for, e.g. an Obj-C property.
    if (e->isGLValue())
      return CGF.buildLoadOfLValue(CGF.buildLValue(e), e->getExprLoc())
          .getScalarVal();
    // Otherwise, calculate and project.
    llvm_unreachable("NYI");
  }

  return Visit(op);
}

// Conversion from bool, integral, or floating-point to integral or
// floating-point. Conversions involving other types are handled elsewhere.
// Conversion to bool is handled elsewhere because that's a comparison against
// zero, not a simple cast. This handles both individual scalars and vectors.
mlir::Value ScalarExprEmitter::buildScalarCast(
    mlir::Value src, QualType srcType, QualType dstType, mlir::Type srcTy,
    mlir::Type dstTy, ScalarConversionOpts opts) {
  assert(!srcType->isMatrixType() && !dstType->isMatrixType() &&
         "Internal error: matrix types not handled by this function.");
  if (mlir::isa<mlir::IntegerType>(srcTy) ||
      mlir::isa<mlir::IntegerType>(dstTy))
    llvm_unreachable("Obsolete code. Don't use mlir::IntegerType with CIR.");

  mlir::Type fullDstTy = dstTy;
  if (mlir::isa<mlir::cir::VectorType>(srcTy) &&
      mlir::isa<mlir::cir::VectorType>(dstTy)) {
    // Use the element types of the vectors to figure out the CastKind.
    srcTy = mlir::dyn_cast<mlir::cir::VectorType>(srcTy).getEltType();
    dstTy = mlir::dyn_cast<mlir::cir::VectorType>(dstTy).getEltType();
  }
  assert(!mlir::isa<mlir::cir::VectorType>(srcTy) &&
         !mlir::isa<mlir::cir::VectorType>(dstTy) &&
         "buildScalarCast given a vector type and a non-vector type");

  std::optional<mlir::cir::CastKind> castKind;

  if (mlir::isa<mlir::cir::BoolType>(srcTy)) {
    if (opts.treatBooleanAsSigned)
      llvm_unreachable("NYI: signed bool");
    if (CGF.getBuilder().isInt(dstTy)) {
      castKind = mlir::cir::CastKind::bool_to_int;
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(dstTy)) {
      castKind = mlir::cir::CastKind::bool_to_float;
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else if (CGF.getBuilder().isInt(srcTy)) {
    if (CGF.getBuilder().isInt(dstTy)) {
      castKind = mlir::cir::CastKind::integral;
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(dstTy)) {
      castKind = mlir::cir::CastKind::int_to_float;
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(srcTy)) {
    if (CGF.getBuilder().isInt(dstTy)) {
      // If we can't recognize overflow as undefined behavior, assume that
      // overflow saturates. This protects against normal optimizations if we
      // are compiling with non-standard FP semantics.
      if (!CGF.cgm.getCodeGenOpts().StrictFloatCastOverflow)
        llvm_unreachable("NYI");
      if (Builder.getIsFPConstrained())
        llvm_unreachable("NYI");
      castKind = mlir::cir::CastKind::float_to_int;
    } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(dstTy)) {
      // TODO: split this to createFPExt/createFPTrunc
      return Builder.createFloatingCast(src, fullDstTy);
    } else {
      llvm_unreachable("Internal error: Cast to unexpected type");
    }
  } else {
    llvm_unreachable("Internal error: Cast from unexpected type");
  }

  assert(castKind.has_value() && "Internal error: CastKind not set.");
  return Builder.create<mlir::cir::CastOp>(src.getLoc(), fullDstTy, *castKind,
                                           src);
}

LValue
CIRGenFunction::buildCompoundAssignmentLValue(const CompoundAssignOperator *E) {
  ScalarExprEmitter Scalar(*this, builder);
  mlir::Value Result;
  switch (E->getOpcode()) {
#define COMPOUND_OP(Op)                                                        \
  case BO_##Op##Assign:                                                        \
    return Scalar.buildCompoundAssignLValue(E, &ScalarExprEmitter::build##Op,  \
                                            Result)
    COMPOUND_OP(Mul);
    COMPOUND_OP(Div);
    COMPOUND_OP(Rem);
    COMPOUND_OP(Add);
    COMPOUND_OP(Sub);
    COMPOUND_OP(Shl);
    COMPOUND_OP(Shr);
    COMPOUND_OP(And);
    COMPOUND_OP(Xor);
    COMPOUND_OP(Or);
#undef COMPOUND_OP

  case BO_PtrMemD:
  case BO_PtrMemI:
  case BO_Mul:
  case BO_Div:
  case BO_Rem:
  case BO_Add:
  case BO_Sub:
  case BO_Shl:
  case BO_Shr:
  case BO_LT:
  case BO_GT:
  case BO_LE:
  case BO_GE:
  case BO_EQ:
  case BO_NE:
  case BO_Cmp:
  case BO_And:
  case BO_Xor:
  case BO_Or:
  case BO_LAnd:
  case BO_LOr:
  case BO_Assign:
  case BO_Comma:
    llvm_unreachable("Not valid compound assignment operators");
  }
  llvm_unreachable("Unhandled compound assignment operator");
}

LValue ScalarExprEmitter::buildCompoundAssignLValue(
    const CompoundAssignOperator *e,
    mlir::Value (ScalarExprEmitter::*func)(const BinOpInfo &),
    mlir::Value &result) {
  QualType lhsTy = e->getLHS()->getType();
  BinOpInfo opInfo;

  if (e->getComputationResultType()->isAnyComplexType())
    assert(0 && "not implemented");

  // Emit the RHS first.  __block variables need to have the rhs evaluated
  // first, plus this should improve codegen a little.

  QualType promotionTypeCr = getPromotionType(e->getComputationResultType());
  if (promotionTypeCr.isNull())
    promotionTypeCr = e->getComputationResultType();

  QualType promotionTypeLhs = getPromotionType(e->getComputationLHSType());
  QualType promotionTypeRhs = getPromotionType(e->getRHS()->getType());

  if (!promotionTypeRhs.isNull())
    opInfo.rhs = CGF.buildPromotedScalarExpr(e->getRHS(), promotionTypeRhs);
  else
    opInfo.rhs = Visit(e->getRHS());

  opInfo.fullType = promotionTypeCr;
  opInfo.compType = opInfo.fullType;
  if (const auto *vecType = dyn_cast_or_null<VectorType>(opInfo.fullType)) {
    opInfo.compType = vecType->getElementType();
  }
  opInfo.opcode = e->getOpcode();
  opInfo.fpFeatures = e->getFPFeaturesInEffect(CGF.getLangOpts());
  opInfo.e = e;
  opInfo.loc = e->getSourceRange();

  // Load/convert the LHS
  LValue lhslv = CGF.buildLValue(e->getLHS());

  if (const AtomicType *atomicTy = lhsTy->getAs<AtomicType>()) {
    assert(0 && "not implemented");
  }

  opInfo.lhs = buildLoadOfLValue(lhslv, e->getExprLoc());

  CIRGenFunction::SourceLocRAIIObject sourceloc{
      CGF, CGF.getLoc(e->getSourceRange())};
  SourceLocation loc = e->getExprLoc();
  if (!promotionTypeLhs.isNull())
    opInfo.lhs = buildScalarConversion(opInfo.lhs, lhsTy, promotionTypeLhs,
                                       e->getExprLoc());
  else
    opInfo.lhs = buildScalarConversion(opInfo.lhs, lhsTy,
                                       e->getComputationLHSType(), loc);

  // Expand the binary operator.
  result = (this->*func)(opInfo);

  // Convert the result back to the LHS type,
  // potentially with Implicit Conversion sanitizer check.
  result = buildScalarConversion(result, promotionTypeCr, lhsTy, loc,
                                 ScalarConversionOpts(CGF.SanOpts));

  // Store the result value into the LHS lvalue. Bit-fields are handled
  // specially because the result is altered by the store, i.e., [C99 6.5.16p1]
  // 'An assignment expression has the value of the left operand after the
  // assignment...'.
  if (lhslv.isBitField())
    CGF.buildStoreThroughBitfieldLValue(RValue::get(result), lhslv, result);
  else
    CGF.buildStoreThroughLValue(RValue::get(result), lhslv);

  if (CGF.getLangOpts().OpenMP)
    CGF.cgm.getOpenMPRuntime().checkAndEmitLastprivateConditional(CGF,
                                                                  e->getLHS());
  return lhslv;
}

mlir::Value ScalarExprEmitter::buildComplexToScalarConversion(
    mlir::Location loc, mlir::Value v, CastKind kind, QualType destTy) {
  mlir::cir::CastKind castOpKind;
  switch (kind) {
  case CK_FloatingComplexToReal:
    castOpKind = mlir::cir::CastKind::float_complex_to_real;
    break;
  case CK_IntegralComplexToReal:
    castOpKind = mlir::cir::CastKind::int_complex_to_real;
    break;
  case CK_FloatingComplexToBoolean:
    castOpKind = mlir::cir::CastKind::float_complex_to_bool;
    break;
  case CK_IntegralComplexToBoolean:
    castOpKind = mlir::cir::CastKind::int_complex_to_bool;
    break;
  default:
    llvm_unreachable("invalid complex-to-scalar cast kind");
  }

  return Builder.createCast(loc, castOpKind, v, CGF.ConvertType(destTy));
}

mlir::Value ScalarExprEmitter::buildNullValue(QualType ty, mlir::Location loc) {
  return CGF.buildFromMemory(CGF.cgm.buildNullConstant(ty, loc), ty);
}

mlir::Value ScalarExprEmitter::buildPromoted(const Expr *e,
                                             QualType PromotionType) {
  e = e->IgnoreParens();
  if (const auto *BO = dyn_cast<BinaryOperator>(e)) {
    switch (BO->getOpcode()) {
#define HANDLE_BINOP(OP)                                                       \
  case BO_##OP:                                                                \
    return build##OP(buildBinOps(BO, PromotionType));
      HANDLE_BINOP(Add)
      HANDLE_BINOP(Sub)
      HANDLE_BINOP(Mul)
      HANDLE_BINOP(Div)
#undef HANDLE_BINOP
    default:
      break;
    }
  } else if (const auto *uo = dyn_cast<UnaryOperator>(e)) {
    switch (uo->getOpcode()) {
    case UO_Imag:
    case UO_Real:
      llvm_unreachable("NYI");
    case UO_Minus:
      return visitMinus(uo, PromotionType);
    case UO_Plus:
      return visitPlus(uo, PromotionType);
    default:
      break;
    }
  }
  auto result = Visit(const_cast<Expr *>(e));
  if (result) {
    if (!PromotionType.isNull())
      return buildPromotedValue(result, PromotionType);
    return buildUnPromotedValue(result, e->getType());
  }
  return result;
}

mlir::Value ScalarExprEmitter::buildCompoundAssign(
    const CompoundAssignOperator *e,
    mlir::Value (ScalarExprEmitter::*func)(const BinOpInfo &)) {

  bool ignore = testAndClearIgnoreResultAssign();
  mlir::Value rhs;
  LValue lhs = buildCompoundAssignLValue(e, func, rhs);

  // If the result is clearly ignored, return now.
  if (ignore)
    return {};

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return rhs;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!lhs.isVolatileQualified())
    return rhs;

  // Otherwise, reload the value.
  return buildLoadOfLValue(lhs, e->getExprLoc());
}

mlir::Value ScalarExprEmitter::VisitExprWithCleanups(ExprWithCleanups *e) {
  auto scopeLoc = CGF.getLoc(e->getSourceRange());
  auto &builder = CGF.builder;

  auto scope = builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Type &yieldTy, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                              builder.getInsertionBlock()};
        auto scopeYieldVal = Visit(e->getSubExpr());
        if (scopeYieldVal) {
          builder.create<mlir::cir::YieldOp>(loc, scopeYieldVal);
          yieldTy = scopeYieldVal.getType();
        }
      });

  // Defend against dominance problems caused by jumps out of expression
  // evaluation through the shared cleanup block.
  // TODO(cir): Scope.ForceCleanup({&V});
  return scope.getNumResults() > 0 ? scope->getResult(0) : nullptr;
}

mlir::Value ScalarExprEmitter::VisitBinAssign(const BinaryOperator *e) {
  bool ignore = testAndClearIgnoreResultAssign();

  mlir::Value rhs;
  LValue lhs;

  switch (e->getLHS()->getType().getObjCLifetime()) {
  case Qualifiers::OCL_Strong:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_Autoreleasing:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_ExplicitNone:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_Weak:
    llvm_unreachable("NYI");
  case Qualifiers::OCL_None:
    // __block variables need to have the rhs evaluated first, plus this should
    // improve codegen just a little.
    rhs = Visit(e->getRHS());
    lhs = buildCheckedLValue(e->getLHS(), CIRGenFunction::TCK_Store);

    // Store the value into the LHS. Bit-fields are handled specially because
    // the result is altered by the store, i.e., [C99 6.5.16p1]
    // 'An assignment expression has the value of the left operand after the
    // assignment...'.
    if (lhs.isBitField()) {
      CGF.buildStoreThroughBitfieldLValue(RValue::get(rhs), lhs, rhs);
    } else {
      CGF.buildNullabilityCheck(lhs, rhs, e->getExprLoc());
      CIRGenFunction::SourceLocRAIIObject loc{CGF,
                                              CGF.getLoc(e->getSourceRange())};
      CGF.buildStoreThroughLValue(RValue::get(rhs), lhs);
    }
  }

  // If the result is clearly ignored, return now.
  if (ignore)
    return nullptr;

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return rhs;

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!lhs.isVolatileQualified())
    return rhs;

  // Otherwise, reload the value.
  return buildLoadOfLValue(lhs, e->getExprLoc());
}

/// Return true if the specified expression is cheap enough and side-effect-free
/// enough to evaluate unconditionally instead of conditionally.  This is used
/// to convert control flow into selects in some cases.
/// TODO(cir): can be shared with LLVM codegen.
static bool isCheapEnoughToEvaluateUnconditionally(const Expr *e,
                                                   CIRGenFunction &cgf) {
  // Anything that is an integer or floating point constant is fine.
  return e->IgnoreParens()->isEvaluatable(cgf.getContext());

  // Even non-volatile automatic variables can't be evaluated unconditionally.
  // Referencing a thread_local may cause non-trivial initialization work to
  // occur. If we're inside a lambda and one of the variables is from the scope
  // outside the lambda, that function may have returned already. Reading its
  // locals is a bad idea. Also, these reads may introduce races there didn't
  // exist in the source-level program.
}

mlir::Value ScalarExprEmitter::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *e) {
  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(e->getSourceRange());
  testAndClearIgnoreResultAssign();

  // Bind the common expression if necessary.
  CIRGenFunction::OpaqueValueMapping binding(CGF, e);

  Expr *condExpr = e->getCond();
  Expr *lhsExpr = e->getTrueExpr();
  Expr *rhsExpr = e->getFalseExpr();

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm.
  bool condExprBool;
  if (CGF.ConstantFoldsToSimpleInteger(condExpr, condExprBool)) {
    Expr *live = lhsExpr, *dead = rhsExpr;
    if (!condExprBool)
      std::swap(live, dead);

    // If the dead side doesn't have labels we need, just emit the Live part.
    if (!CGF.ContainsLabel(dead)) {
      if (condExprBool)
        assert(!MissingFeatures::incrementProfileCounter());
      auto result = Visit(live);

      // If the live part is a throw expression, it acts like it has a void
      // type, so evaluating it returns a null Value.  However, a conditional
      // with non-void type must return a non-null Value.
      if (!result && !e->getType()->isVoidType()) {
        llvm_unreachable("NYI");
      }

      return result;
    }
  }

  // OpenCL: If the condition is a vector, we can treat this condition like
  // the select function.
  if ((CGF.getLangOpts().OpenCL && condExpr->getType()->isVectorType()) ||
      condExpr->getType()->isExtVectorType()) {
    llvm_unreachable("NYI");
  }

  if (condExpr->getType()->isVectorType() ||
      condExpr->getType()->isSveVLSBuiltinType()) {
    assert(condExpr->getType()->isVectorType() && "?: op for SVE vector NYI");
    mlir::Value condValue = Visit(condExpr);
    mlir::Value lhsValue = Visit(lhsExpr);
    mlir::Value rhsValue = Visit(rhsExpr);
    return builder.create<mlir::cir::VecTernaryOp>(loc, condValue, lhsValue,
                                                   rhsValue);
  }

  // If this is a really simple expression (like x ? 4 : 5), emit this as a
  // select instead of as control flow.  We can only do this if it is cheap and
  // safe to evaluate the LHS and RHS unconditionally.
  if (isCheapEnoughToEvaluateUnconditionally(lhsExpr, CGF) &&
      isCheapEnoughToEvaluateUnconditionally(rhsExpr, CGF)) {
    bool lhsIsVoid = false;
    auto condV = CGF.evaluateExprAsBool(condExpr);
    assert(!MissingFeatures::incrementProfileCounter());

    return builder
        .create<mlir::cir::TernaryOp>(
            loc, condV, /*thenBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto lhs = Visit(lhsExpr);
              if (!lhs) {
                lhs = builder.getNullValue(CGF.VoidTy, loc);
                lhsIsVoid = true;
              }
              builder.create<mlir::cir::YieldOp>(loc, lhs);
            },
            /*elseBuilder=*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              auto rhs = Visit(rhsExpr);
              if (lhsIsVoid) {
                assert(!rhs && "lhs and rhs types must match");
                rhs = builder.getNullValue(CGF.VoidTy, loc);
              }
              builder.create<mlir::cir::YieldOp>(loc, rhs);
            })
        .getResult();
  }

  mlir::Value condV = CGF.buildOpOnBoolExpr(loc, condExpr);
  CIRGenFunction::ConditionalEvaluation eval(CGF);
  SmallVector<mlir::OpBuilder::InsertPoint, 2> insertPoints{};
  mlir::Type yieldTy{};

  auto patchVoidOrThrowSites = [&]() {
    if (insertPoints.empty())
      return;
    // If both arms are void, so be it.
    if (!yieldTy)
      yieldTy = CGF.VoidTy;

    // Insert required yields.
    for (auto &toInsert : insertPoints) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.restoreInsertionPoint(toInsert);

      // Block does not return: build empty yield.
      if (mlir::isa<mlir::cir::VoidType>(yieldTy)) {
        builder.create<mlir::cir::YieldOp>(loc);
      } else { // Block returns: set null yield value.
        mlir::Value op0 = builder.getNullValue(yieldTy, loc);
        builder.create<mlir::cir::YieldOp>(loc, op0);
      }
    }
  };

  return builder
      .create<mlir::cir::TernaryOp>(
          loc, condV, /*trueBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                  b.getInsertionBlock()};
            CGF.currLexScope->setAsTernary();

            assert(!MissingFeatures::incrementProfileCounter());
            eval.begin(CGF);
            auto lhs = Visit(lhsExpr);
            eval.end(CGF);

            if (lhs) {
              yieldTy = lhs.getType();
              b.create<mlir::cir::YieldOp>(loc, lhs);
              return;
            }
            // If LHS or RHS is a throw or void expression we need to patch arms
            // as to properly match yield types.
            insertPoints.push_back(b.saveInsertionPoint());
          },
          /*falseBuilder=*/
          [&](mlir::OpBuilder &b, mlir::Location loc) {
            CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                  b.getInsertionBlock()};
            CGF.currLexScope->setAsTernary();

            assert(!MissingFeatures::incrementProfileCounter());
            eval.begin(CGF);
            auto rhs = Visit(rhsExpr);
            eval.end(CGF);

            if (rhs) {
              yieldTy = rhs.getType();
              b.create<mlir::cir::YieldOp>(loc, rhs);
            } else {
              // If LHS or RHS is a throw or void expression we need to patch
              // arms as to properly match yield types.
              insertPoints.push_back(b.saveInsertionPoint());
            }

            patchVoidOrThrowSites();
          })
      .getResult();
}

mlir::Value CIRGenFunction::buildScalarPrePostIncDec(const UnaryOperator *e,
                                                     LValue lv, bool isInc,
                                                     bool isPre) {
  return ScalarExprEmitter(*this, builder)
      .buildScalarPrePostIncDec(e, lv, isInc, isPre);
}

mlir::Value ScalarExprEmitter::VisitBinLAnd(const clang::BinaryOperator *e) {
  if (e->getType()->isVectorType()) {
    llvm_unreachable("NYI");
  }

  bool instrumentRegions = CGF.cgm.getCodeGenOpts().hasProfileClangInstr();
  mlir::Type resTy = convertType(e->getType());
  mlir::Location loc = CGF.getLoc(e->getExprLoc());

  // If we have 0 && RHS, see if we can elide RHS, if so, just return 0.
  // If we have 1 && X, just emit X without inserting the control flow.
  bool lhsCondVal;
  if (CGF.ConstantFoldsToSimpleInteger(e->getLHS(), lhsCondVal)) {
    if (lhsCondVal) { // If we have 1 && X, just emit X.

      mlir::Value rhsCond = CGF.evaluateExprAsBool(e->getRHS());

      if (instrumentRegions) {
        llvm_unreachable("NYI");
      }
      // ZExt result to int or bool.
      return Builder.createZExtOrBitCast(rhsCond.getLoc(), rhsCond, resTy);
    }
    // 0 && RHS: If it is safe, just elide the RHS, and return 0/false.
    if (!CGF.ContainsLabel(e->getRHS()))
      return Builder.getNullValue(resTy, loc);
  }

  CIRGenFunction::ConditionalEvaluation eval(CGF);

  mlir::Value lhsCondV = CGF.evaluateExprAsBool(e->getLHS());
  auto resOp = Builder.create<mlir::cir::TernaryOp>(
      loc, lhsCondV, /*trueBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc, b.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        mlir::Value rhsCondV = CGF.evaluateExprAsBool(e->getRHS());
        auto res = b.create<mlir::cir::TernaryOp>(
            loc, rhsCondV, /*trueBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                    b.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<mlir::cir::ConstantOp>(
                  loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       true));
              b.create<mlir::cir::YieldOp>(loc, res.getRes());
            },
            /*falseBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                    b.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<mlir::cir::ConstantOp>(
                  loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       false));
              b.create<mlir::cir::YieldOp>(loc, res.getRes());
            });
        b.create<mlir::cir::YieldOp>(loc, res.getResult());
      },
      /*falseBuilder*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc, b.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        auto res = b.create<mlir::cir::ConstantOp>(
            loc, Builder.getBoolTy(),
            Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(), false));
        b.create<mlir::cir::YieldOp>(loc, res.getRes());
      });
  return Builder.createZExtOrBitCast(resOp.getLoc(), resOp.getResult(), resTy);
}

mlir::Value ScalarExprEmitter::VisitBinLOr(const clang::BinaryOperator *e) {
  if (e->getType()->isVectorType()) {
    llvm_unreachable("NYI");
  }

  bool instrumentRegions = CGF.cgm.getCodeGenOpts().hasProfileClangInstr();
  mlir::Type resTy = convertType(e->getType());
  mlir::Location Loc = CGF.getLoc(e->getExprLoc());

  // If we have 1 || RHS, see if we can elide RHS, if so, just return 1.
  // If we have 0 || X, just emit X without inserting the control flow.
  bool lhsCondVal;
  if (CGF.ConstantFoldsToSimpleInteger(e->getLHS(), lhsCondVal)) {
    if (!lhsCondVal) { // If we have 0 || X, just emit X.

      mlir::Value rhsCond = CGF.evaluateExprAsBool(e->getRHS());

      if (instrumentRegions) {
        llvm_unreachable("NYI");
      }
      // ZExt result to int or bool.
      return Builder.createZExtOrBitCast(rhsCond.getLoc(), rhsCond, resTy);
    }
    // 1 || RHS: If it is safe, just elide the RHS, and return 1/true.
    if (!CGF.ContainsLabel(e->getRHS())) {
      if (auto intTy = mlir::dyn_cast<mlir::cir::IntType>(resTy))
        return Builder.getConstInt(Loc, intTy, 1);
      return Builder.getBool(true, Loc);
    }
  }

  CIRGenFunction::ConditionalEvaluation eval(CGF);

  mlir::Value lhsCondV = CGF.evaluateExprAsBool(e->getLHS());
  auto resOp = Builder.create<mlir::cir::TernaryOp>(
      Loc, lhsCondV, /*trueBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc, b.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        auto res = b.create<mlir::cir::ConstantOp>(
            loc, Builder.getBoolTy(),
            Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(), true));
        b.create<mlir::cir::YieldOp>(loc, res.getRes());
      },
      /*falseBuilder*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        CIRGenFunction::LexicalScope lexScope{CGF, loc, b.getInsertionBlock()};
        CGF.currLexScope->setAsTernary();
        mlir::Value rhsCondV = CGF.evaluateExprAsBool(e->getRHS());
        auto res = b.create<mlir::cir::TernaryOp>(
            loc, rhsCondV, /*trueBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              SmallVector<mlir::Location, 2> locs;
              if (mlir::isa<mlir::FileLineColLoc>(loc)) {
                locs.push_back(loc);
                locs.push_back(loc);
              } else if (mlir::isa<mlir::FusedLoc>(loc)) {
                auto fusedLoc = mlir::cast<mlir::FusedLoc>(loc);
                locs.push_back(fusedLoc.getLocations()[0]);
                locs.push_back(fusedLoc.getLocations()[1]);
              }
              CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                    b.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<mlir::cir::ConstantOp>(
                  loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       true));
              b.create<mlir::cir::YieldOp>(loc, res.getRes());
            },
            /*falseBuilder*/
            [&](mlir::OpBuilder &b, mlir::Location loc) {
              SmallVector<mlir::Location, 2> locs;
              if (mlir::isa<mlir::FileLineColLoc>(loc)) {
                locs.push_back(loc);
                locs.push_back(loc);
              } else if (mlir::isa<mlir::FusedLoc>(loc)) {
                auto fusedLoc = mlir::cast<mlir::FusedLoc>(loc);
                locs.push_back(fusedLoc.getLocations()[0]);
                locs.push_back(fusedLoc.getLocations()[1]);
              }
              CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                    b.getInsertionBlock()};
              CGF.currLexScope->setAsTernary();
              auto res = b.create<mlir::cir::ConstantOp>(
                  loc, Builder.getBoolTy(),
                  Builder.getAttr<mlir::cir::BoolAttr>(Builder.getBoolTy(),
                                                       false));
              b.create<mlir::cir::YieldOp>(loc, res.getRes());
            });
        b.create<mlir::cir::YieldOp>(loc, res.getResult());
      });

  return Builder.createZExtOrBitCast(resOp.getLoc(), resOp.getResult(), resTy);
}

mlir::Value ScalarExprEmitter::VisitVAArgExpr(VAArgExpr *ve) {
  QualType ty = ve->getType();

  if (ty->isVariablyModifiedType())
    assert(!MissingFeatures::variablyModifiedTypeEmission() && "NYI");

  Address argValue = Address::invalid();
  mlir::Value val = CGF.buildVAArg(ve, argValue);

  return val;
}

/// Return the size or alignment of the type of argument of the sizeof
/// expression as an integer.
mlir::Value ScalarExprEmitter::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *e) {
  QualType typeToSize = e->getTypeOfArgument();
  if (e->getKind() == UETT_SizeOf) {
    if (const VariableArrayType *vat =
            CGF.getContext().getAsVariableArrayType(typeToSize)) {

      if (e->isArgumentType()) {
        // sizeof(type) - make sure to emit the VLA size.
        CGF.buildVariablyModifiedType(typeToSize);
      } else {
        // C99 6.5.3.4p2: If the argument is an expression of type
        // VLA, it is evaluated.
        CGF.buildIgnoredExpr(e->getArgumentExpr());
      }

      auto vlaSize = CGF.getVLASize(vat);
      mlir::Value size = vlaSize.NumElts;

      // Scale the number of non-VLA elements by the non-VLA element size.
      CharUnits eltSize = CGF.getContext().getTypeSizeInChars(vlaSize.Type);
      if (!eltSize.isOne())
        size = Builder.createMul(size, CGF.cgm.getSize(eltSize).getValue());

      return size;
    }
  } else if (e->getKind() == UETT_OpenMPRequiredSimdAlign) {
    llvm_unreachable("NYI");
  }

  // If this isn't sizeof(vla), the result must be constant; use the constant
  // folding logic so we don't have to duplicate it here.
  return Builder.getConstInt(CGF.getLoc(e->getSourceRange()),
                             e->EvaluateKnownConstInt(CGF.getContext()));
}

mlir::Value CIRGenFunction::buildCheckedInBoundsGEP(
    mlir::Type elemTy, mlir::Value ptr, ArrayRef<mlir::Value> idxList,
    bool signedIndices, bool isSubtraction, SourceLocation loc) {
  mlir::Type ptrTy = ptr.getType();
  assert(idxList.size() == 1 && "multi-index ptr arithmetic NYI");
  mlir::Value gepVal = builder.create<mlir::cir::PtrStrideOp>(
      cgm.getLoc(loc), ptrTy, ptr, idxList[0]);

  // If the pointer overflow sanitizer isn't enabled, do nothing.
  if (!SanOpts.has(SanitizerKind::PointerOverflow))
    return gepVal;

  // TODO(cir): the unreachable code below hides a substantial amount of code
  // from the original codegen related with pointer overflow sanitizer.
  assert(MissingFeatures::pointerOverflowSanitizer());
  llvm_unreachable("pointer overflow sanitizer NYI");
}
