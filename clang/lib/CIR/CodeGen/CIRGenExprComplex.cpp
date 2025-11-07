#include "CIRGenBuilder.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"

#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace clang::CIRGen;

#ifndef NDEBUG
/// Return the complex type that we are meant to emit.
static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type))
    return comp;
  return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
}
#endif // NDEBUG

namespace {

class ComplexExprEmitter : public StmtVisitor<ComplexExprEmitter, mlir::Value> {
  CIRGenFunction &cgf;
  CIRGenBuilderTy &builder;
  bool fpHasBeenPromoted = false;

public:
  explicit ComplexExprEmitter(CIRGenFunction &cgf)
      : cgf(cgf), builder(cgf.getBuilder()) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// Given an expression with complex type that represents a value l-value,
  /// this method emits the address of the l-value, then loads and returns the
  /// result.
  mlir::Value emitLoadOfLValue(const Expr *e) {
    return emitLoadOfLValue(cgf.emitLValue(e), e->getExprLoc());
  }

  mlir::Value emitLoadOfLValue(LValue lv, SourceLocation loc);

  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void emitStoreOfComplex(mlir::Location loc, mlir::Value val, LValue lv,
                          bool isInit);

  /// Emit a cast from complex value Val to DestType.
  mlir::Value emitComplexToComplexCast(mlir::Value val, QualType srcType,
                                       QualType destType, SourceLocation loc);

  /// Emit a cast from scalar value Val to DestType.
  mlir::Value emitScalarToComplexCast(mlir::Value val, QualType srcType,
                                      QualType destType, SourceLocation loc);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *e) {
    assert(!cir::MissingFeatures::generateDebugInfo());
    return StmtVisitor<ComplexExprEmitter, mlir::Value>::Visit(e);
  }

  mlir::Value VisitStmt(Stmt *s) {
    s->dump(llvm::errs(), cgf.getContext());
    llvm_unreachable("Stmt can't have complex result type!");
  }

  mlir::Value VisitExpr(Expr *s);
  mlir::Value VisitConstantExpr(ConstantExpr *e) {
    if (mlir::Attribute result = ConstantEmitter(cgf).tryEmitConstantExpr(e))
      return builder.getConstant(cgf.getLoc(e->getSourceRange()),
                                 mlir::cast<mlir::TypedAttr>(result));
    return Visit(e->getSubExpr());
  }

  mlir::Value VisitParenExpr(ParenExpr *pe) { return Visit(pe->getSubExpr()); }
  mlir::Value VisitGenericSelectionExpr(GenericSelectionExpr *ge) {
    return Visit(ge->getResultExpr());
  }
  mlir::Value VisitImaginaryLiteral(const ImaginaryLiteral *il);
  mlir::Value
  VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *pe) {
    return Visit(pe->getReplacement());
  }
  mlir::Value VisitCoawaitExpr(CoawaitExpr *s) {
    llvm_unreachable("VisitCoawaitExpr NYI");
  }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *s) {
    llvm_unreachable("VisitCoyieldExpr NYI");
  }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *e) {
    return Visit(e->getSubExpr());
  }

  mlir::Value emitConstant(const CIRGenFunction::ConstantEmission &constant,
                           Expr *e) {
    assert(constant && "not a constant");
    if (constant.isReference())
      return emitLoadOfLValue(constant.getReferenceLValue(cgf, e),
                              e->getExprLoc());

    mlir::TypedAttr valueAttr = constant.getValue();
    return builder.getConstant(cgf.getLoc(e->getSourceRange()), valueAttr);
  }

  // l-values.
  mlir::Value VisitDeclRefExpr(DeclRefExpr *e) {
    if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(e))
      return emitConstant(constant, e);
    return emitLoadOfLValue(e);
  }
  mlir::Value VisitObjCIvarRefExpr(ObjCIvarRefExpr *e) {
    llvm_unreachable("VisitObjCIvarRefExpr NYI");
  }
  mlir::Value VisitObjCMessageExpr(ObjCMessageExpr *e) {
    llvm_unreachable("VisitObjCMessageExpr NYI");
  }
  mlir::Value VisitArraySubscriptExpr(Expr *e) { return emitLoadOfLValue(e); }

  mlir::Value VisitMemberExpr(MemberExpr *me) {
    if (CIRGenFunction::ConstantEmission constant = cgf.tryEmitAsConstant(me)) {
      cgf.emitIgnoredExpr(me->getBase());
      return emitConstant(constant, me);
    }
    return emitLoadOfLValue(me);
  }

  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *e) {
    if (e->isGLValue())
      return emitLoadOfLValue(cgf.getOrCreateOpaqueLValueMapping(e),
                              e->getExprLoc());

    // Otherwise, assume the mapping is the scalar directly.
    return cgf.getOrCreateOpaqueRValueMapping(e).getScalarVal();
  }

  mlir::Value VisitPseudoObjectExpr(PseudoObjectExpr *e) {
    llvm_unreachable("VisitPseudoObjectExpr NYI");
  }

  mlir::Value emitCast(CastKind ck, Expr *op, QualType destTy);
  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *e) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    if (e->changesVolatileQualification())
      return emitLoadOfLValue(e);
    return emitCast(e->getCastKind(), e->getSubExpr(), e->getType());
  }
  mlir::Value VisitCastExpr(CastExpr *e) {
    if (const auto *ece = dyn_cast<ExplicitCastExpr>(e))
      cgf.CGM.emitExplicitCastExprType(ece, &cgf);
    if (e->changesVolatileQualification())
      return emitLoadOfLValue(e);
    return emitCast(e->getCastKind(), e->getSubExpr(), e->getType());
  }
  mlir::Value VisitCallExpr(const CallExpr *e);
  mlir::Value VisitStmtExpr(const StmtExpr *e);

  // Operators.
  mlir::Value VisitPrePostIncDec(const UnaryOperator *e, bool isInc,
                                 bool isPre) {
    LValue lv = cgf.emitLValue(e->getSubExpr());
    return cgf.emitComplexPrePostIncDec(e, lv, isInc, isPre);
  }
  mlir::Value VisitUnaryPostDec(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, /*isInc=*/false, /*isPre=*/false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, /*isInc=*/true, /*isPre=*/false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, /*isInc=*/false, /*isPre=*/true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *e) {
    return VisitPrePostIncDec(e, /*isInc=*/true, /*isPre=*/true);
  }
  mlir::Value VisitUnaryDeref(const Expr *e) { return emitLoadOfLValue(e); }

  mlir::Value VisitUnaryPlus(const UnaryOperator *e,
                             QualType promotionType = QualType());
  mlir::Value VisitPlus(const UnaryOperator *e, QualType promotionType);
  mlir::Value VisitUnaryMinus(const UnaryOperator *e,
                              QualType promotionType = QualType());
  mlir::Value VisitMinus(const UnaryOperator *e, QualType promotionType);
  mlir::Value VisitUnaryNot(const UnaryOperator *e);
  // LNot,Real,Imag never return complex.
  mlir::Value VisitUnaryExtension(const UnaryOperator *e) {
    return Visit(e->getSubExpr());
  }
  mlir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *dae) {
    CIRGenFunction::CXXDefaultArgExprScope scope(cgf, dae);
    return Visit(dae->getExpr());
  }

  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    CIRGenFunction::CXXDefaultInitExprScope scope(cgf, die);
    return Visit(die->getExpr());
  }

  mlir::Value VisitExprWithCleanups(ExprWithCleanups *e) {
    CIRGenFunction::RunCleanupsScope scope(cgf);
    mlir::Value V = Visit(e->getSubExpr());
    // Defend against dominance problems caused by jumps out of expression
    // evaluation through the shared cleanup block.
    scope.ForceCleanup({&V});
    return V;
  }
  mlir::Value VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *e) {
    mlir::Location loc = cgf.getLoc(e->getExprLoc());
    mlir::Type complexTy = cgf.convertType(e->getType());
    return builder.getNullValue(complexTy, loc);
  }
  mlir::Value VisitImplicitValueInitExpr(ImplicitValueInitExpr *e) {
    llvm_unreachable("VisitImplicitValueInitExpr NYI");
  }

  struct BinOpInfo {
    mlir::Location loc;
    mlir::Value lhs{};
    mlir::Value rhs{};
    QualType ty{}; // Computation Type.
    FPOptions fpFeatures{};
  };

  BinOpInfo emitBinOps(const BinaryOperator *e,
                       QualType promotionTy = QualType());
  mlir::Value emitPromoted(const Expr *e, QualType promotionTy);
  mlir::Value emitPromotedComplexOperand(const Expr *e, QualType promotionTy);
  LValue emitCompoundAssignLValue(
      const CompoundAssignOperator *e,
      mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &), RValue &val);
  mlir::Value emitCompoundAssign(
      const CompoundAssignOperator *e,
      mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &));

  mlir::Value emitBinAdd(const BinOpInfo &op);
  mlir::Value emitBinSub(const BinOpInfo &op);
  mlir::Value emitBinMul(const BinOpInfo &op);
  mlir::Value emitBinDiv(const BinOpInfo &op);

  QualType higherPrecisionTypeForComplexArithmetic(QualType elementType,
                                                   bool isDivOpCode) {
    ASTContext &astContext = cgf.getContext();
    const QualType higherElementType =
        astContext.GetHigherPrecisionFPType(elementType);
    const llvm::fltSemantics &elementTypeSemantics =
        astContext.getFloatTypeSemantics(elementType);
    const llvm::fltSemantics &higherElementTypeSemantics =
        astContext.getFloatTypeSemantics(higherElementType);
    // Check that the promoted type can handle the intermediate values without
    // overflowing. This can be interpreted as:
    // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal) * 2 <=
    // LargerType.LargestFiniteVal.
    // In terms of exponent it gives this formula:
    // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal
    // doubles the exponent of SmallerType.LargestFiniteVal)
    if (llvm::APFloat::semanticsMaxExponent(elementTypeSemantics) * 2 + 1 <=
        llvm::APFloat::semanticsMaxExponent(higherElementTypeSemantics)) {
      fpHasBeenPromoted = true;
      return astContext.getComplexType(higherElementType);
    }

    // The intermediate values can't be represented in the promoted type
    // without overflowing.
    return QualType();
  }

  QualType getPromotionType(QualType rt, bool isDivOpCode = false) {
    if (auto *ct = rt->getAs<ComplexType>()) {
      QualType elementType = ct->getElementType();
      if (isDivOpCode && elementType->isFloatingType() &&
          cgf.getLangOpts().getComplexRange() ==
              LangOptions::ComplexRangeKind::CX_Promoted)
        return higherPrecisionTypeForComplexArithmetic(elementType,
                                                       isDivOpCode);
      if (elementType.UseExcessPrecision(cgf.getContext()))
        return cgf.getContext().getComplexType(cgf.getContext().FloatTy);
    }

    return rt.UseExcessPrecision(cgf.getContext()) ? cgf.getContext().FloatTy
                                                   : QualType();
  }

#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *e) {                          \
    QualType promotionTy = getPromotionType(                                   \
        e->getType(), e->getOpcode() == BinaryOperatorKind::BO_Div);           \
    mlir::Value result = emitBin##OP(emitBinOps(e, promotionTy));              \
    if (!promotionTy.isNull())                                                 \
      result = cgf.emitUnPromotedValue(result, e->getType());                  \
    return result;                                                             \
  }

  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
#undef HANDLEBINOP

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *e) {
    llvm_unreachable("VisitCXXRewrittenBinaryOperator NYI");
  }

  // Compound assignments.
  mlir::Value VisitBinAddAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinAdd);
  }
  mlir::Value VisitBinSubAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinSub);
  }
  mlir::Value VisitBinMulAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinMul);
  }
  mlir::Value VisitBinDivAssign(const CompoundAssignOperator *e) {
    return emitCompoundAssign(e, &ComplexExprEmitter::emitBinDiv);
  }

  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.

  LValue emitBinAssignLValue(const BinaryOperator *e, mlir::Value &val);
  mlir::Value VisitBinAssign(const BinaryOperator *e);
  mlir::Value VisitBinComma(const BinaryOperator *e);

  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *e);
  mlir::Value VisitChooseExpr(ChooseExpr *ce);

  mlir::Value VisitInitListExpr(InitListExpr *e);

  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
    return emitLoadOfLValue(e);
  }

  mlir::Value VisitVAArgExpr(VAArgExpr *e);

  mlir::Value VisitAtomicExpr(AtomicExpr *e) {
    return cgf.emitAtomicExpr(e).getComplexVal();
  }

  mlir::Value VisitPackIndexingExpr(PackIndexingExpr *e) {
    llvm_unreachable("VisitPackIndexingExpr NYI");
  }
};

} // namespace

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitLoadOfLValue - Given an RValue reference for a complex, emit code to
/// load the real and imaginary pieces, returning them as Real/Imag.
mlir::Value ComplexExprEmitter::emitLoadOfLValue(LValue lv,
                                                 SourceLocation loc) {
  assert(lv.isSimple() && "non-simple complex l-value?");
  if (lv.getType()->isAtomicType())
    llvm_unreachable("emitLoadOfLValue AtomicType NYI");

  Address srcPtr = lv.getAddress();
  return builder.createLoad(cgf.getLoc(loc), srcPtr, lv.isVolatileQualified());
}

/// EmitStoreOfComplex - Store the specified real/imag parts into the
/// specified value pointer.
void ComplexExprEmitter::emitStoreOfComplex(mlir::Location loc, mlir::Value val,
                                            LValue lv, bool isInit) {
  if (lv.getType()->isAtomicType() ||
      (!isInit && cgf.LValueIsSuitableForInlineAtomic(lv)))
    llvm_unreachable("emitStoreOfComplex AtomicType NYI");

  Address destAddr = lv.getAddress();
  builder.createStore(loc, val, destAddr, lv.isVolatileQualified());
}

static mlir::Value createComplexFromReal(CIRGenBuilderTy &builder,
                                         mlir::Location loc, mlir::Value real) {
  mlir::Value imag = builder.getNullValue(real.getType(), loc);
  return builder.createComplexCreate(loc, real, imag);
}

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

mlir::Value ComplexExprEmitter::VisitExpr(Expr *s) {
  llvm_unreachable("not supported");
}

mlir::Value
ComplexExprEmitter::VisitImaginaryLiteral(const ImaginaryLiteral *il) {
  mlir::Location loc = cgf.getLoc(il->getExprLoc());
  auto ty = mlir::cast<cir::ComplexType>(cgf.convertType(il->getType()));
  mlir::Type elementTy = ty.getElementType();

  mlir::TypedAttr realValueAttr;
  mlir::TypedAttr imagValueAttr;
  if (mlir::isa<cir::IntType>(elementTy)) {
    auto imagValue = cast<IntegerLiteral>(il->getSubExpr())->getValue();
    realValueAttr = cir::IntAttr::get(elementTy, 0);
    imagValueAttr = cir::IntAttr::get(elementTy, imagValue);
  } else if (mlir::isa<cir::FPTypeInterface>(elementTy)) {
    auto imagValue = cast<FloatingLiteral>(il->getSubExpr())->getValue();
    realValueAttr = cir::FPAttr::get(
        elementTy, llvm::APFloat::getZero(imagValue.getSemantics()));
    imagValueAttr = cir::FPAttr::get(elementTy, imagValue);
  } else {
    llvm_unreachable("unexpected complex element type");
  }

  auto realValue = builder.getConstant(loc, realValueAttr);
  auto imagValue = builder.getConstant(loc, imagValueAttr);
  return builder.createComplexCreate(loc, realValue, imagValue);
}

mlir::Value ComplexExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(cgf.getContext())->isReferenceType())
    return emitLoadOfLValue(e);
  return cgf.emitCallExpr(e).getComplexVal();
}

mlir::Value ComplexExprEmitter::VisitStmtExpr(const StmtExpr *e) {
  llvm_unreachable("VisitStmtExpr NYI");
}

mlir::Value ComplexExprEmitter::emitComplexToComplexCast(mlir::Value val,
                                                         QualType srcType,
                                                         QualType destType,
                                                         SourceLocation loc) {
  if (srcType == destType)
    return val;

  // Get the src/dest element type.
  QualType srcElemTy = srcType->castAs<ComplexType>()->getElementType();
  QualType destElemTy = destType->castAs<ComplexType>()->getElementType();

  cir::CastKind castOpKind;
  if (srcElemTy->isFloatingType() && destElemTy->isFloatingType())
    castOpKind = cir::CastKind::float_complex;
  else if (srcElemTy->isFloatingType() && destElemTy->isIntegerType())
    castOpKind = cir::CastKind::float_complex_to_int_complex;
  else if (srcElemTy->isIntegerType() && destElemTy->isFloatingType())
    castOpKind = cir::CastKind::int_complex_to_float_complex;
  else if (srcElemTy->isIntegerType() && destElemTy->isIntegerType())
    castOpKind = cir::CastKind::int_complex;
  else
    llvm_unreachable("unexpected src type or dest type");

  return builder.createCast(cgf.getLoc(loc), castOpKind, val,
                            cgf.convertType(destType));
}

mlir::Value ComplexExprEmitter::emitScalarToComplexCast(mlir::Value vVal,
                                                        QualType srcType,
                                                        QualType destType,
                                                        SourceLocation loc) {
  cir::CastKind castOpKind;
  if (srcType->isFloatingType())
    castOpKind = cir::CastKind::float_to_complex;
  else if (srcType->isIntegerType())
    castOpKind = cir::CastKind::int_to_complex;
  else
    llvm_unreachable("unexpected src type");

  return builder.createCast(cgf.getLoc(loc), castOpKind, vVal,
                            cgf.convertType(destType));
}

mlir::Value ComplexExprEmitter::emitCast(CastKind ck, Expr *op,
                                         QualType destTy) {
  switch (ck) {
  case CK_Dependent:
    llvm_unreachable("dependent cast kind in IR gen!");

  case CK_NoOp:
  case CK_LValueToRValue:
  case CK_UserDefinedConversion:
    return Visit(op);

  // Atomic to non-atomic casts may be more than a no-op for some platforms
  // and for some types.
  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
    llvm_unreachable("emitCast Atomic NYI");

  case CK_LValueBitCast: {
    LValue origLV = cgf.emitLValue(op);
    Address addr =
        origLV.getAddress().withElementType(builder, cgf.convertType(destTy));
    LValue destLV = cgf.makeAddrLValue(addr, destTy);
    return emitLoadOfLValue(destLV, op->getExprLoc());
  }

  case CK_LValueToRValueBitCast: {
    LValue sourceLVal = cgf.emitLValue(op);
    Address addr = sourceLVal.getAddress().withElementType(
        builder, cgf.convertTypeForMem(destTy));
    LValue destLV = cgf.makeAddrLValue(addr, destTy);
    destLV.setTBAAInfo(TBAAAccessInfo::getMayAliasInfo());
    return emitLoadOfLValue(destLV, op->getExprLoc());
  }

  case CK_BitCast:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_UncheckedDerivedToBase:
  case CK_Dynamic:
  case CK_ToUnion:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
  case CK_ConstructorConversion:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_BooleanToSignedIntegral:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_CPointerToObjCPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_BuiltinFnToFnPtr:
  case CK_ZeroToOCLOpaqueType:
  case CK_AddressSpaceConversion:
  case CK_IntToOCLSampler:
  case CK_FloatingToFixedPoint:
  case CK_FixedPointToFloating:
  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToIntegral:
  case CK_IntegralToFixedPoint:
  case CK_MatrixCast:
  case CK_HLSLVectorTruncation:
  case CK_HLSLArrayRValue:
  case CK_HLSLElementwiseCast:
  case CK_HLSLAggregateSplatCast:
    llvm_unreachable("invalid cast kind for complex value");

  case CK_FloatingRealToComplex:
  case CK_IntegralRealToComplex: {
    assert(!cir::MissingFeatures::CGFPOptionsRAII());
    return emitScalarToComplexCast(cgf.emitScalarExpr(op), op->getType(),
                                   destTy, op->getExprLoc());
  }

  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex: {
    assert(!cir::MissingFeatures::CGFPOptionsRAII());
    return emitComplexToComplexCast(Visit(op), op->getType(), destTy,
                                    op->getExprLoc());
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
}

mlir::Value ComplexExprEmitter::VisitUnaryPlus(const UnaryOperator *e,
                                               QualType promotionType) {
  QualType promotionTy = promotionType.isNull()
                             ? getPromotionType(e->getSubExpr()->getType())
                             : promotionType;
  mlir::Value result = VisitPlus(e, promotionTy);
  return promotionTy.isNull()
             ? result
             : cgf.emitUnPromotedValue(result, e->getSubExpr()->getType());
}

mlir::Value ComplexExprEmitter::VisitPlus(const UnaryOperator *e,
                                          QualType promotionType) {
  mlir::Value op =
      promotionType.isNull()
          ? Visit(e->getSubExpr())
          : cgf.emitPromotedComplexExpr(e->getSubExpr(), promotionType);
  return builder.createUnaryOp(cgf.getLoc(e->getExprLoc()),
                               cir::UnaryOpKind::Plus, op);
}

mlir::Value ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *e,
                                                QualType promotionType) {
  QualType promotionTy = promotionType.isNull()
                             ? getPromotionType(e->getSubExpr()->getType())
                             : promotionType;
  mlir::Value result = VisitMinus(e, promotionTy);
  return promotionTy.isNull()
             ? result
             : cgf.emitUnPromotedValue(result, e->getSubExpr()->getType());
}

mlir::Value ComplexExprEmitter::VisitMinus(const UnaryOperator *e,
                                           QualType promotionType) {
  mlir::Value op;
  if (!promotionType.isNull())
    op = cgf.emitPromotedComplexExpr(e->getSubExpr(), promotionType);
  else
    op = Visit(e->getSubExpr());
  return builder.createUnaryOp(cgf.getLoc(e->getExprLoc()),
                               cir::UnaryOpKind::Minus, op);
}

mlir::Value ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *e) {
  mlir::Value op = Visit(e->getSubExpr());
  return builder.createUnaryOp(cgf.getLoc(e->getExprLoc()),
                               cir::UnaryOpKind::Not, op);
}

mlir::Value ComplexExprEmitter::emitBinAdd(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::CGFPOptionsRAII());
  return builder.createComplexAdd(op.loc, op.lhs, op.rhs);
}

mlir::Value ComplexExprEmitter::emitBinSub(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::CGFPOptionsRAII());
  return builder.createComplexSub(op.loc, op.lhs, op.rhs);
}

static cir::ComplexRangeKind
getComplexRangeAttr(LangOptions::ComplexRangeKind range) {
  switch (range) {
  case LangOptions::CX_Full:
    return cir::ComplexRangeKind::Full;
  case LangOptions::CX_Improved:
    return cir::ComplexRangeKind::Improved;
  case LangOptions::CX_Promoted:
    return cir::ComplexRangeKind::Promoted;
  case LangOptions::CX_Basic:
    return cir::ComplexRangeKind::Basic;
  case LangOptions::CX_None:
    // The default value for ComplexRangeKind is Full if no option is selected
    return cir::ComplexRangeKind::Full;
  }
}

mlir::Value ComplexExprEmitter::emitBinMul(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::CGFPOptionsRAII());
  return builder.createComplexMul(
      op.loc, op.lhs, op.rhs,
      getComplexRangeAttr(op.fpFeatures.getComplexRange()), fpHasBeenPromoted);
}

mlir::Value ComplexExprEmitter::emitBinDiv(const BinOpInfo &op) {
  assert(!cir::MissingFeatures::CGFPOptionsRAII());
  return builder.createComplexDiv(
      op.loc, op.lhs, op.rhs,
      getComplexRangeAttr(op.fpFeatures.getComplexRange()), fpHasBeenPromoted);
}

mlir::Value CIRGenFunction::emitUnPromotedValue(mlir::Value result,
                                                QualType unPromotionType) {
  assert(!mlir::cast<cir::ComplexType>(result.getType()).isIntegerComplex() &&
         "integral complex will never be promoted");
  return builder.createCast(cir::CastKind::float_complex, result,
                            convertType(unPromotionType));
}

mlir::Value CIRGenFunction::emitPromotedValue(mlir::Value result,
                                              QualType promotionType) {
  assert(!mlir::cast<cir::ComplexType>(result.getType()).isIntegerComplex() &&
         "integral complex will never be promoted");
  return builder.createCast(cir::CastKind::float_complex, result,
                            convertType(promotionType));
}

mlir::Value ComplexExprEmitter::emitPromoted(const Expr *e,
                                             QualType PromotionTy) {
  e = e->IgnoreParens();
  if (const auto *bo = dyn_cast<BinaryOperator>(e)) {
    switch (bo->getOpcode()) {
#define HANDLE_BINOP(OP)                                                       \
  case BO_##OP:                                                                \
    return emitBin##OP(emitBinOps(bo, PromotionTy));
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
    case UO_Minus:
      return VisitMinus(uo, PromotionTy);
    case UO_Plus:
      return VisitPlus(uo, PromotionTy);
    default:
      break;
    }
  }

  mlir::Value result = Visit(const_cast<Expr *>(e));
  if (!PromotionTy.isNull())
    return cgf.emitPromotedValue(result, PromotionTy);
  return result;
}

mlir::Value CIRGenFunction::emitPromotedComplexExpr(const Expr *e,
                                                    QualType promotionType) {
  return ComplexExprEmitter(*this).emitPromoted(e, promotionType);
}

mlir::Value
ComplexExprEmitter::emitPromotedComplexOperand(const Expr *e,
                                               QualType promotionTy) {
  if (e->getType()->isAnyComplexType()) {
    if (!promotionTy.isNull())
      return cgf.emitPromotedComplexExpr(e, promotionTy);
    return Visit(const_cast<Expr *>(e));
  }

  mlir::Value real;
  if (!promotionTy.isNull()) {
    QualType complexElementTy =
        promotionTy->castAs<ComplexType>()->getElementType();
    real = cgf.emitPromotedScalarExpr(e, complexElementTy);
  } else
    real = cgf.emitScalarExpr(e);
  return createComplexFromReal(builder, cgf.getLoc(e->getExprLoc()), real);
}

ComplexExprEmitter::BinOpInfo
ComplexExprEmitter::emitBinOps(const BinaryOperator *e, QualType promotionTy) {
  BinOpInfo ops{cgf.getLoc(e->getExprLoc())};
  ops.lhs = emitPromotedComplexOperand(e->getLHS(), promotionTy);
  ops.rhs = emitPromotedComplexOperand(e->getRHS(), promotionTy);
  if (!promotionTy.isNull())
    ops.ty = promotionTy;
  else
    ops.ty = e->getType();
  ops.fpFeatures = e->getFPFeaturesInEffect(cgf.getLangOpts());
  return ops;
}

LValue ComplexExprEmitter::emitCompoundAssignLValue(
    const CompoundAssignOperator *e,
    mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &), RValue &val) {
  QualType lhsTy = e->getLHS()->getType();
  if (const AtomicType *at = lhsTy->getAs<AtomicType>())
    lhsTy = at->getValueType();

  BinOpInfo opInfo{cgf.getLoc(e->getExprLoc())};
  opInfo.fpFeatures = e->getFPFeaturesInEffect(cgf.getLangOpts());

  assert(!cir::MissingFeatures::CGFPOptionsRAII());

  // Load the RHS and LHS operands.
  // __block variables need to have the rhs evaluated first, plus this should
  // improve codegen a little.
  QualType promotionTypeCR;
  promotionTypeCR = getPromotionType(e->getComputationResultType());
  if (promotionTypeCR.isNull())
    promotionTypeCR = e->getComputationResultType();
  opInfo.ty = promotionTypeCR;
  QualType complexElementTy =
      opInfo.ty->castAs<ComplexType>()->getElementType();
  QualType promotionTypeRHS = getPromotionType(e->getRHS()->getType());

  // The RHS should have been converted to the computation type.
  if (e->getRHS()->getType()->isRealFloatingType()) {
    if (!promotionTypeRHS.isNull())
      opInfo.rhs = createComplexFromReal(
          builder, cgf.getLoc(e->getExprLoc()),
          cgf.emitPromotedScalarExpr(e->getRHS(), promotionTypeRHS));
    else {
      assert(cgf.getContext().hasSameUnqualifiedType(complexElementTy,
                                                     e->getRHS()->getType()));
      opInfo.rhs = createComplexFromReal(builder, cgf.getLoc(e->getExprLoc()),
                                         cgf.emitScalarExpr(e->getRHS()));
    }
  } else {
    if (!promotionTypeRHS.isNull()) {
      opInfo.rhs = cgf.emitPromotedComplexExpr(e->getRHS(), promotionTypeRHS);
    } else {
      assert(cgf.getContext().hasSameUnqualifiedType(opInfo.ty,
                                                     e->getRHS()->getType()));
      opInfo.rhs = Visit(e->getRHS());
    }
  }

  LValue lhs = cgf.emitLValue(e->getLHS());

  // Load from the l-value and convert it.
  SourceLocation loc = e->getExprLoc();
  QualType promotionTypeLHS = getPromotionType(e->getComputationLHSType());
  if (lhsTy->isAnyComplexType()) {
    mlir::Value lhsVal = emitLoadOfLValue(lhs, loc);
    if (!promotionTypeLHS.isNull())
      opInfo.lhs =
          emitComplexToComplexCast(lhsVal, lhsTy, promotionTypeLHS, loc);
    else
      opInfo.lhs = emitComplexToComplexCast(lhsVal, lhsTy, opInfo.ty, loc);
  } else {
    mlir::Value lhsVal = cgf.emitLoadOfScalar(lhs, loc);
    // For floating point real operands we can directly pass the scalar form
    // to the binary operator emission and potentially get more efficient code.
    if (lhsTy->isRealFloatingType()) {
      QualType promotedComplexElementTy;
      if (!promotionTypeLHS.isNull()) {
        promotedComplexElementTy =
            cast<ComplexType>(promotionTypeLHS)->getElementType();
        if (!cgf.getContext().hasSameUnqualifiedType(promotedComplexElementTy,
                                                     promotionTypeLHS))
          lhsVal = cgf.emitScalarConversion(lhsVal, lhsTy,
                                            promotedComplexElementTy, loc);
      } else {
        if (!cgf.getContext().hasSameUnqualifiedType(complexElementTy, lhsTy))
          lhsVal =
              cgf.emitScalarConversion(lhsVal, lhsTy, complexElementTy, loc);
      }
      opInfo.lhs =
          createComplexFromReal(builder, cgf.getLoc(e->getExprLoc()), lhsVal);
    } else {
      opInfo.lhs = emitScalarToComplexCast(lhsVal, lhsTy, opInfo.ty, loc);
    }
  }

  // Expand the binary operator.
  mlir::Value result = (this->*func)(opInfo);

  // Truncate the result and store it into the LHS lvalue.
  if (lhsTy->isAnyComplexType()) {
    mlir::Value resVal =
        emitComplexToComplexCast(result, opInfo.ty, lhsTy, loc);
    emitStoreOfComplex(cgf.getLoc(e->getExprLoc()), resVal, lhs,
                       /*isInit=*/false);
    val = RValue::getComplex(resVal);
  } else {
    mlir::Value resVal =
        cgf.emitComplexToScalarConversion(result, opInfo.ty, lhsTy, loc);
    cgf.emitStoreOfScalar(resVal, lhs, /*isInit=*/false);
    val = RValue::get(resVal);
  }

  return lhs;
}

mlir::Value ComplexExprEmitter::emitCompoundAssign(
    const CompoundAssignOperator *e,
    mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &)) {
  RValue val;
  LValue lv = emitCompoundAssignLValue(e, func, val);

  // The result of an assignment in C is the assigned r-value.
  if (!cgf.getLangOpts().CPlusPlus)
    return val.getComplexVal();

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!lv.isVolatileQualified())
    return val.getComplexVal();

  return emitLoadOfLValue(lv, e->getExprLoc());
}

LValue ComplexExprEmitter::emitBinAssignLValue(const BinaryOperator *e,
                                               mlir::Value &val) {
  assert(cgf.getContext().hasSameUnqualifiedType(e->getLHS()->getType(),
                                                 e->getRHS()->getType()) &&
         "Invalid assignment");

  // Emit the RHS.  __block variables need the RHS evaluated first.
  val = Visit(e->getRHS());

  // Compute the address to store into.
  LValue lhs = cgf.emitLValue(e->getLHS());

  // Store the result value into the LHS lvalue.
  emitStoreOfComplex(cgf.getLoc(e->getExprLoc()), val, lhs, /*isInit=*/false);
  return lhs;
}

mlir::Value ComplexExprEmitter::VisitBinAssign(const BinaryOperator *e) {
  mlir::Value val;
  LValue lv = emitBinAssignLValue(e, val);

  // The result of an assignment in C is the assigned r-value.
  if (!cgf.getLangOpts().CPlusPlus)
    return val;

  // If the lvalue is non-volatile, return the computed value of the
  // assignment.
  if (!lv.isVolatileQualified())
    return val;

  return emitLoadOfLValue(lv, e->getExprLoc());
}

mlir::Value ComplexExprEmitter::VisitBinComma(const BinaryOperator *e) {
  cgf.emitIgnoredExpr(e->getLHS());
  return Visit(e->getRHS());
}

mlir::Value ComplexExprEmitter::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *e) {
  mlir::Location loc = cgf.getLoc(e->getSourceRange());

  // Bind the common expression if necessary.
  CIRGenFunction::OpaqueValueMapping binding(cgf, e);

  CIRGenFunction::ConditionalEvaluation eval(cgf);

  Expr *cond = e->getCond()->IgnoreParens();
  mlir::Value condValue = cgf.evaluateExprAsBool(cond);

  return cir::TernaryOp::create(
             builder, loc, condValue,
             /*trueBuilder=*/
             [&](mlir::OpBuilder &b, mlir::Location loc) {
               eval.begin(cgf);
               mlir::Value trueValue = Visit(e->getTrueExpr());
               cir::YieldOp::create(b, loc, trueValue);
               eval.end(cgf);
             },
             /*falseBuilder=*/
             [&](mlir::OpBuilder &b, mlir::Location loc) {
               eval.begin(cgf);
               mlir::Value falseValue = Visit(e->getFalseExpr());
               cir::YieldOp::create(b, loc, falseValue);
               eval.end(cgf);
             })
      .getResult();
}

mlir::Value ComplexExprEmitter::VisitChooseExpr(ChooseExpr *ce) {
  return Visit(ce->getChosenSubExpr());
}

mlir::Value ComplexExprEmitter::VisitInitListExpr(InitListExpr *e) {
  mlir::Location loc = cgf.getLoc(e->getExprLoc());
  if (e->getNumInits() == 2) {
    mlir::Value real = cgf.emitScalarExpr(e->getInit(0));
    mlir::Value imag = cgf.emitScalarExpr(e->getInit(1));
    return builder.createComplexCreate(loc, real, imag);
  }

  if (e->getNumInits() == 1)
    return Visit(e->getInit(0));

  assert(e->getNumInits() == 0 && "Unexpected number of inits");
  mlir::Type complexTy = cgf.convertType(e->getType());
  return builder.getNullValue(complexTy, loc);
}

mlir::Value ComplexExprEmitter::VisitVAArgExpr(VAArgExpr *e) {
  llvm_unreachable("VisitVAArgExpr NYI");
}

//===----------------------------------------------------------------------===//
//                         Entry Point into this File
//===----------------------------------------------------------------------===//

mlir::Value CIRGenFunction::emitComplexExpr(const Expr *e) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");
  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::emitComplexExprIntoLValue(const Expr *e, LValue dest,
                                               bool isInit) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");
  ComplexExprEmitter emitter(*this);
  mlir::Value val = emitter.Visit(const_cast<Expr *>(e));
  emitter.emitStoreOfComplex(getLoc(e->getExprLoc()), val, dest, isInit);
}

/// EmitStoreOfComplex - Store a complex number into the specified l-value.
void CIRGenFunction::emitStoreOfComplex(mlir::Location loc, mlir::Value v,
                                        LValue dest, bool isInit) {
  ComplexExprEmitter(*this).emitStoreOfComplex(loc, v, dest, isInit);
}

mlir::Value CIRGenFunction::emitLoadOfComplex(LValue src, SourceLocation loc) {
  return ComplexExprEmitter(*this).emitLoadOfLValue(src, loc);
}

LValue CIRGenFunction::emitComplexAssignmentLValue(const BinaryOperator *e) {
  assert(e->getOpcode() == BO_Assign);
  mlir::Value val; // ignored
  LValue lVal = ComplexExprEmitter(*this).emitBinAssignLValue(e, val);
  if (getLangOpts().OpenMP)
    llvm_unreachable("emitComplexAssignmentLValue: OpenMP NYI");
  return lVal;
}

using CompoundFunc =
    mlir::Value (ComplexExprEmitter::*)(const ComplexExprEmitter::BinOpInfo &);

static CompoundFunc getComplexOp(BinaryOperatorKind op) {
  switch (op) {
  case BO_MulAssign:
    return &ComplexExprEmitter::emitBinMul;
  case BO_DivAssign:
    return &ComplexExprEmitter::emitBinDiv;
  case BO_SubAssign:
    return &ComplexExprEmitter::emitBinSub;
  case BO_AddAssign:
    return &ComplexExprEmitter::emitBinAdd;
  default:
    llvm_unreachable("unexpected complex compound assignment");
  }
}

LValue CIRGenFunction::emitComplexCompoundAssignmentLValue(
    const CompoundAssignOperator *e) {
  CompoundFunc op = getComplexOp(e->getOpcode());
  RValue val;
  return ComplexExprEmitter(*this).emitCompoundAssignLValue(e, op, val);
}

mlir::Value CIRGenFunction::emitComplexPrePostIncDec(const UnaryOperator *e,
                                                     LValue lv, bool isInc,
                                                     bool isPre) {
  mlir::Value inVal = emitLoadOfComplex(lv, e->getExprLoc());
  mlir::Location loc = getLoc(e->getExprLoc());
  auto opKind = isInc ? cir::UnaryOpKind::Inc : cir::UnaryOpKind::Dec;
  mlir::Value incVal = builder.createUnaryOp(loc, opKind, inVal);

  // Store the updated result through the lvalue.
  emitStoreOfComplex(loc, incVal, lv, /*isInit=*/false);
  if (getLangOpts().OpenMP)
    llvm_unreachable("emitComplexPrePostIncDec: OpenMP NYI");

  // If this is a post inc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? incVal : inVal;
}

LValue CIRGenFunction::emitScalarCompoundAssignWithComplex(
    const CompoundAssignOperator *e, mlir::Value &result) {
  CompoundFunc op = getComplexOp(e->getOpcode());
  RValue val;
  LValue ret = ComplexExprEmitter(*this).emitCompoundAssignLValue(e, op, val);
  result = val.getScalarVal();
  return ret;
}
