#include "CIRGenBuilder.h"
#include "CIRGenCstEmitter.h"
#include "CIRGenFunction.h"
#include "clang/Basic/LangOptions.h"
#include "clang/CIR/Interfaces/CIRFPTypeInterface.h"
#include "clang/CIR/MissingFeatures.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/ErrorHandling.h"

using namespace cir;
using namespace clang;

namespace {

class ComplexExprEmitter : public StmtVisitor<ComplexExprEmitter, mlir::Value> {
  CIRGenFunction &CGF;
  CIRGenBuilderTy &builder;
  bool FPHasBeenPromoted = false;

public:
  explicit ComplexExprEmitter(CIRGenFunction &cgf)
      : CGF(cgf), builder(cgf.getBuilder()) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// Given an expression with complex type that represents a value l-value,
  /// this method emits the address of the l-value, then loads and returns the
  /// result.
  mlir::Value buildLoadOfLValue(const Expr *e) {
    return buildLoadOfLValue(CGF.buildLValue(e), e->getExprLoc());
  }

  mlir::Value buildLoadOfLValue(LValue lv, SourceLocation loc);

  /// EmitStoreOfComplex - Store the specified real/imag parts into the
  /// specified value pointer.
  void buildStoreOfComplex(mlir::Location loc, mlir::Value val, LValue lv,
                           bool isInit);

  /// Emit a cast from complex value Val to DestType.
  mlir::Value buildComplexToComplexCast(mlir::Value val, QualType srcType,
                                        QualType destType, SourceLocation loc);
  /// Emit a cast from scalar value Val to DestType.
  mlir::Value buildScalarToComplexCast(mlir::Value val, QualType srcType,
                                       QualType destType, SourceLocation loc);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  mlir::Value Visit(Expr *e) {
    assert(!MissingFeatures::generateDebugInfo());
    return StmtVisitor<ComplexExprEmitter, mlir::Value>::Visit(e);
  }

  mlir::Value VisitStmt(Stmt *s) {
    s->dump(llvm::errs(), CGF.getContext());
    llvm_unreachable("Stmt can't have complex result type!");
  }

  mlir::Value VisitExpr(Expr *s) { llvm_unreachable("not supported"); }
  mlir::Value VisitConstantExpr(ConstantExpr *e) {
    if (auto result = ConstantEmitter(CGF).tryEmitConstantExpr(e))
      return builder.getConstant(CGF.getLoc(e->getSourceRange()),
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
  mlir::Value VisitCoawaitExpr(CoawaitExpr *s) { llvm_unreachable("NYI"); }
  mlir::Value VisitCoyieldExpr(CoyieldExpr *s) { llvm_unreachable("NYI"); }
  mlir::Value VisitUnaryCoawait(const UnaryOperator *e) {
    return Visit(e->getSubExpr());
  }

  mlir::Value emitConstant(const CIRGenFunction::ConstantEmission &constant,
                           Expr *e) {
    assert(constant && "not a constant");
    if (constant.isReference())
      return buildLoadOfLValue(constant.getReferenceLValue(CGF, e),
                               e->getExprLoc());

    auto valueAttr = constant.getValue();
    return builder.getConstant(CGF.getLoc(e->getSourceRange()), valueAttr);
  }

  // l-values.
  mlir::Value VisitDeclRefExpr(DeclRefExpr *e) {
    if (CIRGenFunction::ConstantEmission constant = CGF.tryEmitAsConstant(e))
      return emitConstant(constant, e);
    return buildLoadOfLValue(e);
  }
  mlir::Value VisitObjCIvarRefExpr(ObjCIvarRefExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitObjCMessageExpr(ObjCMessageExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitArraySubscriptExpr(Expr *e) { llvm_unreachable("NYI"); }
  mlir::Value VisitMemberExpr(MemberExpr *me) { llvm_unreachable("NYI"); }
  mlir::Value VisitOpaqueValueExpr(OpaqueValueExpr *e) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitPseudoObjectExpr(PseudoObjectExpr *e) {
    llvm_unreachable("NYI");
  }

  // FIXME: CompoundLiteralExpr

  mlir::Value buildCast(CastKind ck, Expr *op, QualType destTy);
  mlir::Value VisitImplicitCastExpr(ImplicitCastExpr *e) {
    // Unlike for scalars, we don't have to worry about function->ptr demotion
    // here.
    if (e->changesVolatileQualification())
      return buildLoadOfLValue(e);
    return buildCast(e->getCastKind(), e->getSubExpr(), e->getType());
  }
  mlir::Value VisitCastExpr(CastExpr *e);
  mlir::Value VisitCallExpr(const CallExpr *e);
  mlir::Value VisitStmtExpr(const StmtExpr *e) { llvm_unreachable("NYI"); }

  // Operators.
  mlir::Value visitPrePostIncDec(const UnaryOperator *e, bool isInc,
                                 bool isPre);
  mlir::Value VisitUnaryPostDec(const UnaryOperator *e) {
    return visitPrePostIncDec(e, false, false);
  }
  mlir::Value VisitUnaryPostInc(const UnaryOperator *e) {
    return visitPrePostIncDec(e, true, false);
  }
  mlir::Value VisitUnaryPreDec(const UnaryOperator *e) {
    return visitPrePostIncDec(e, false, true);
  }
  mlir::Value VisitUnaryPreInc(const UnaryOperator *e) {
    return visitPrePostIncDec(e, true, true);
  }
  mlir::Value VisitUnaryDeref(const Expr *e) { llvm_unreachable("NYI"); }

  mlir::Value VisitUnaryPlus(const UnaryOperator *e,
                             QualType promotionType = QualType());
  mlir::Value visitPlus(const UnaryOperator *e, QualType promotionType);
  mlir::Value VisitUnaryMinus(const UnaryOperator *e,
                              QualType promotionType = QualType());
  mlir::Value visitMinus(const UnaryOperator *e, QualType promotionType);
  mlir::Value VisitUnaryNot(const UnaryOperator *e);
  // LNot,Real,Imag never return complex.
  mlir::Value VisitUnaryExtension(const UnaryOperator *e) {
    return Visit(e->getSubExpr());
  }
  mlir::Value VisitCXXDefaultArgExpr(CXXDefaultArgExpr *dae) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitExprWithCleanups(ExprWithCleanups *e) {
    CIRGenFunction::RunCleanupsScope scope(CGF);
    mlir::Value v = Visit(e->getSubExpr());
    // Defend against dominance problems caused by jumps out of expression
    // evaluation through the shared cleanup block.
    scope.ForceCleanup({&v});
    return v;
  }
  mlir::Value VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *e) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitImplicitValueInitExpr(ImplicitValueInitExpr *e) {
    llvm_unreachable("NYI");
  }

  struct BinOpInfo {
    mlir::Location loc;
    mlir::Value lhs{};
    mlir::Value rhs{};
    QualType ty{}; // Computation Type.
    FPOptions fpFeatures{};
  };

  BinOpInfo buildBinOps(const BinaryOperator *e,
                        QualType promotionTy = QualType());
  mlir::Value buildPromoted(const Expr *e, QualType promotionTy);
  mlir::Value buildPromotedComplexOperand(const Expr *e, QualType promotionTy);

  LValue buildCompoundAssignLValue(
      const CompoundAssignOperator *e,
      mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &), RValue &val);
  mlir::Value buildCompoundAssign(
      const CompoundAssignOperator *e,
      mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &));

  mlir::Value buildBinAdd(const BinOpInfo &op);
  mlir::Value buildBinSub(const BinOpInfo &op);
  mlir::Value buildBinMul(const BinOpInfo &op);
  mlir::Value buildBinDiv(const BinOpInfo &op);

  QualType getHigherPrecisionFpType(QualType elementType) {
    const auto *currentBt = cast<BuiltinType>(elementType);
    switch (currentBt->getKind()) {
    case BuiltinType::Kind::Float16:
      return CGF.getContext().FloatTy;
    case BuiltinType::Kind::Float:
    case BuiltinType::Kind::BFloat16:
      return CGF.getContext().DoubleTy;
    case BuiltinType::Kind::Double:
      return CGF.getContext().LongDoubleTy;
    default:
      return elementType;
    }
  }

  QualType higherPrecisionTypeForComplexArithmetic(QualType elementType,
                                                   bool isDivOpCode) {
    QualType higherElementType = getHigherPrecisionFpType(elementType);
    const llvm::fltSemantics &elementTypeSemantics =
        CGF.getContext().getFloatTypeSemantics(elementType);
    const llvm::fltSemantics &higherElementTypeSemantics =
        CGF.getContext().getFloatTypeSemantics(higherElementType);
    // Check that the promoted type can handle the intermediate values without
    // overflowing. This can be interpreted as:
    // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal) * 2 <=
    // LargerType.LargestFiniteVal.
    // In terms of exponent it gives this formula:
    // (SmallerType.LargestFiniteVal * SmallerType.LargestFiniteVal
    // doubles the exponent of SmallerType.LargestFiniteVal)
    if (llvm::APFloat::semanticsMaxExponent(elementTypeSemantics) * 2 + 1 <=
        llvm::APFloat::semanticsMaxExponent(higherElementTypeSemantics)) {
      FPHasBeenPromoted = true;
      return CGF.getContext().getComplexType(higherElementType);
    }

    DiagnosticsEngine &diags = CGF.CGM.getDiags();
    diags.Report(diag::warn_next_larger_fp_type_same_size_than_fp);
    return QualType();
  }

  QualType getPromotionType(QualType ty, bool isDivOpCode = false) {
    if (auto *ct = ty->getAs<ComplexType>()) {
      QualType elementType = ct->getElementType();
      if (isDivOpCode && elementType->isFloatingType() &&
          CGF.getLangOpts().getComplexRange() ==
              LangOptions::ComplexRangeKind::CX_Promoted)
        return higherPrecisionTypeForComplexArithmetic(elementType,
                                                       isDivOpCode);
      if (elementType.UseExcessPrecision(CGF.getContext()))
        return CGF.getContext().getComplexType(CGF.getContext().FloatTy);
    }
    if (ty.UseExcessPrecision(CGF.getContext()))
      return CGF.getContext().FloatTy;
    return QualType();
  }

#define HANDLEBINOP(OP)                                                        \
  mlir::Value VisitBin##OP(const BinaryOperator *E) {                          \
    QualType promotionTy = getPromotionType(                                   \
        E->getType(),                                                          \
        (E->getOpcode() == BinaryOperatorKind::BO_Div) ? true : false);        \
    mlir::Value result = buildBin##OP(buildBinOps(E, promotionTy));            \
    if (!promotionTy.isNull())                                                 \
      result = CGF.buildUnPromotedValue(result, E->getType());                 \
    return result;                                                             \
  }

  HANDLEBINOP(Mul)
  HANDLEBINOP(Div)
  HANDLEBINOP(Add)
  HANDLEBINOP(Sub)
#undef HANDLEBINOP

  mlir::Value VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *e) {
    llvm_unreachable("NYI");
  }

  // Compound assignments.
  mlir::Value VisitBinAddAssign(const CompoundAssignOperator *e) {
    return buildCompoundAssign(e, &ComplexExprEmitter::buildBinAdd);
  }
  mlir::Value VisitBinSubAssign(const CompoundAssignOperator *e) {
    return buildCompoundAssign(e, &ComplexExprEmitter::buildBinSub);
  }
  mlir::Value VisitBinMulAssign(const CompoundAssignOperator *e) {
    return buildCompoundAssign(e, &ComplexExprEmitter::buildBinMul);
  }
  mlir::Value VisitBinDivAssign(const CompoundAssignOperator *e) {
    return buildCompoundAssign(e, &ComplexExprEmitter::buildBinDiv);
  }

  // GCC rejects rem/and/or/xor for integer complex.
  // Logical and/or always return int, never complex.

  // No comparisons produce a complex result.

  LValue buildBinAssignLValue(const BinaryOperator *e, mlir::Value &val);
  mlir::Value VisitBinAssign(const BinaryOperator *e) {
    mlir::Value val;
    LValue lv = buildBinAssignLValue(e, val);

    // The result of an assignment in C is the assigned r-value.
    if (!CGF.getLangOpts().CPlusPlus)
      return val;

    // If the lvalue is non-volatile, return the computed value of the
    // assignment.
    if (!lv.isVolatileQualified())
      return val;

    return buildLoadOfLValue(lv, e->getExprLoc());
  };
  mlir::Value VisitBinComma(const BinaryOperator *e) {
    llvm_unreachable("NYI");
  }

  mlir::Value
  VisitAbstractConditionalOperator(const AbstractConditionalOperator *co) {
    llvm_unreachable("NYI");
  }
  mlir::Value VisitChooseExpr(ChooseExpr *ce) { llvm_unreachable("NYI"); }

  mlir::Value VisitInitListExpr(InitListExpr *e);

  mlir::Value VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
    llvm_unreachable("NYI");
  }

  mlir::Value VisitVAArgExpr(VAArgExpr *e) { llvm_unreachable("NYI"); }

  mlir::Value VisitAtomicExpr(AtomicExpr *e) { llvm_unreachable("NYI"); }

  mlir::Value VisitPackIndexingExpr(PackIndexingExpr *e) {
    llvm_unreachable("NYI");
  }
};

} // namespace

static const ComplexType *getComplexType(QualType type) {
  type = type.getCanonicalType();
  if (const ComplexType *comp = dyn_cast<ComplexType>(type))
    return comp;
  return cast<ComplexType>(cast<AtomicType>(type)->getValueType());
}

static mlir::Value createComplexFromReal(CIRGenBuilderTy &builder,
                                         mlir::Location loc, mlir::Value real) {
  mlir::Value imag = builder.getNullValue(real.getType(), loc);
  return builder.createComplexCreate(loc, real, imag);
}

mlir::Value ComplexExprEmitter::buildLoadOfLValue(LValue lv,
                                                  SourceLocation loc) {
  assert(lv.isSimple() && "non-simple complex l-value?");
  if (lv.getType()->isAtomicType())
    llvm_unreachable("NYI");

  Address srcPtr = lv.getAddress();
  return builder.createLoad(CGF.getLoc(loc), srcPtr, lv.isVolatileQualified());
}

void ComplexExprEmitter::buildStoreOfComplex(mlir::Location loc,
                                             mlir::Value val, LValue lv,
                                             bool isInit) {
  if (lv.getType()->isAtomicType() ||
      (!isInit && CGF.LValueIsSuitableForInlineAtomic(lv)))
    llvm_unreachable("NYI");

  Address destAddr = lv.getAddress();
  builder.createStore(loc, val, destAddr, lv.isVolatileQualified());
}

mlir::Value ComplexExprEmitter::buildComplexToComplexCast(mlir::Value val,
                                                          QualType srcType,
                                                          QualType destType,
                                                          SourceLocation loc) {
  if (srcType == destType)
    return val;

  // Get the src/dest element type.
  QualType srcElemTy = srcType->castAs<ComplexType>()->getElementType();
  QualType destElemTy = destType->castAs<ComplexType>()->getElementType();

  mlir::cir::CastKind castOpKind;
  if (srcElemTy->isFloatingType() && destElemTy->isFloatingType())
    castOpKind = mlir::cir::CastKind::float_complex;
  else if (srcElemTy->isFloatingType() && destElemTy->isIntegerType())
    castOpKind = mlir::cir::CastKind::float_complex_to_int_complex;
  else if (srcElemTy->isIntegerType() && destElemTy->isFloatingType())
    castOpKind = mlir::cir::CastKind::int_complex_to_float_complex;
  else if (srcElemTy->isIntegerType() && destElemTy->isIntegerType())
    castOpKind = mlir::cir::CastKind::int_complex;
  else
    llvm_unreachable("unexpected src type or dest type");

  return builder.createCast(CGF.getLoc(loc), castOpKind, val,
                            CGF.ConvertType(destType));
}

mlir::Value ComplexExprEmitter::buildScalarToComplexCast(mlir::Value val,
                                                         QualType srcType,
                                                         QualType destType,
                                                         SourceLocation loc) {
  mlir::cir::CastKind castOpKind;
  if (srcType->isFloatingType())
    castOpKind = mlir::cir::CastKind::float_to_complex;
  else if (srcType->isIntegerType())
    castOpKind = mlir::cir::CastKind::int_to_complex;
  else
    llvm_unreachable("unexpected src type");

  return builder.createCast(CGF.getLoc(loc), castOpKind, val,
                            CGF.ConvertType(destType));
}

mlir::Value ComplexExprEmitter::buildCast(CastKind ck, Expr *op,
                                          QualType destTy) {
  switch (ck) {
  case CK_Dependent:
    llvm_unreachable("dependent cast kind in IR gen!");

  // Atomic to non-atomic casts may be more than a no-op for some platforms and
  // for some types.
  case CK_LValueToRValue:
    return Visit(op);

  case CK_AtomicToNonAtomic:
  case CK_NonAtomicToAtomic:
  case CK_NoOp:
  case CK_UserDefinedConversion:
    llvm_unreachable("NYI");

  case CK_LValueBitCast:
    llvm_unreachable("NYI");

  case CK_LValueToRValueBitCast:
    llvm_unreachable("NYI");

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
    llvm_unreachable("invalid cast kind for complex value");

  case CK_FloatingRealToComplex:
  case CK_IntegralRealToComplex: {
    assert(!MissingFeatures::CGFPOptionsRAII());
    return buildScalarToComplexCast(CGF.buildScalarExpr(op), op->getType(),
                                    destTy, op->getExprLoc());
  }

  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex: {
    assert(!MissingFeatures::CGFPOptionsRAII());
    return buildComplexToComplexCast(Visit(op), op->getType(), destTy,
                                     op->getExprLoc());
  }
  }

  llvm_unreachable("unknown cast resulting in complex value");
}

mlir::Value ComplexExprEmitter::VisitCastExpr(CastExpr *e) {
  if (const auto *ece = dyn_cast<ExplicitCastExpr>(e))
    CGF.CGM.buildExplicitCastExprType(ece, &CGF);
  if (e->changesVolatileQualification())
    return buildLoadOfLValue(e);
  return buildCast(e->getCastKind(), e->getSubExpr(), e->getType());
}

mlir::Value ComplexExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(CGF.getContext())->isReferenceType())
    return buildLoadOfLValue(e);

  return CGF.buildCallExpr(e).getComplexVal();
}

mlir::Value ComplexExprEmitter::visitPrePostIncDec(const UnaryOperator *e,
                                                   bool isInc, bool isPre) {
  LValue lv = CGF.buildLValue(e->getSubExpr());
  return CGF.buildComplexPrePostIncDec(e, lv, isInc, isPre);
}

mlir::Value ComplexExprEmitter::VisitUnaryPlus(const UnaryOperator *e,
                                               QualType promotionType) {
  QualType promotionTy = promotionType.isNull()
                             ? getPromotionType(e->getSubExpr()->getType())
                             : promotionType;
  mlir::Value result = visitPlus(e, promotionTy);
  if (!promotionTy.isNull())
    return CGF.buildUnPromotedValue(result, e->getSubExpr()->getType());
  return result;
}

mlir::Value ComplexExprEmitter::visitPlus(const UnaryOperator *e,
                                          QualType promotionType) {
  mlir::Value op;
  if (!promotionType.isNull())
    op = CGF.buildPromotedComplexExpr(e->getSubExpr(), promotionType);
  else
    op = Visit(e->getSubExpr());

  return builder.createUnaryOp(CGF.getLoc(e->getExprLoc()),
                               mlir::cir::UnaryOpKind::Plus, op);
}

mlir::Value ComplexExprEmitter::VisitUnaryMinus(const UnaryOperator *e,
                                                QualType promotionType) {
  QualType promotionTy = promotionType.isNull()
                             ? getPromotionType(e->getSubExpr()->getType())
                             : promotionType;
  mlir::Value result = visitMinus(e, promotionTy);
  if (!promotionTy.isNull())
    return CGF.buildUnPromotedValue(result, e->getSubExpr()->getType());
  return result;
}

mlir::Value ComplexExprEmitter::visitMinus(const UnaryOperator *e,
                                           QualType promotionType) {
  mlir::Value op;
  if (!promotionType.isNull())
    op = CGF.buildPromotedComplexExpr(e->getSubExpr(), promotionType);
  else
    op = Visit(e->getSubExpr());

  return builder.createUnaryOp(CGF.getLoc(e->getExprLoc()),
                               mlir::cir::UnaryOpKind::Minus, op);
}

mlir::Value ComplexExprEmitter::VisitUnaryNot(const UnaryOperator *e) {
  mlir::Value op = Visit(e->getSubExpr());
  return builder.createUnaryOp(CGF.getLoc(e->getExprLoc()),
                               mlir::cir::UnaryOpKind::Not, op);
}

ComplexExprEmitter::BinOpInfo
ComplexExprEmitter::buildBinOps(const BinaryOperator *e, QualType promotionTy) {
  BinOpInfo ops{CGF.getLoc(e->getExprLoc())};

  ops.lhs = buildPromotedComplexOperand(e->getLHS(), promotionTy);
  ops.rhs = buildPromotedComplexOperand(e->getRHS(), promotionTy);
  if (!promotionTy.isNull())
    ops.ty = promotionTy;
  else
    ops.ty = e->getType();
  ops.fpFeatures = e->getFPFeaturesInEffect(CGF.getLangOpts());
  return ops;
}

mlir::Value ComplexExprEmitter::buildPromoted(const Expr *e,
                                              QualType PromotionTy) {
  e = e->IgnoreParens();
  if (const auto *BO = dyn_cast<BinaryOperator>(e)) {
    switch (BO->getOpcode()) {
#define HANDLE_BINOP(OP)                                                       \
  case BO_##OP:                                                                \
    return buildBin##OP(buildBinOps(BO, PromotionTy));
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
      return visitMinus(uo, PromotionTy);
    case UO_Plus:
      return visitPlus(uo, PromotionTy);
    default:
      break;
    }
  }
  auto result = Visit(const_cast<Expr *>(e));
  if (!PromotionTy.isNull())
    return CGF.buildPromotedValue(result, PromotionTy);
  return result;
}

mlir::Value
ComplexExprEmitter::buildPromotedComplexOperand(const Expr *e,
                                                QualType promotionTy) {
  if (e->getType()->isAnyComplexType()) {
    if (!promotionTy.isNull())
      return CGF.buildPromotedComplexExpr(e, promotionTy);
    return Visit(const_cast<Expr *>(e));
  }

  mlir::Value real;
  if (!promotionTy.isNull()) {
    QualType complexElementTy =
        promotionTy->castAs<ComplexType>()->getElementType();
    real = CGF.buildPromotedScalarExpr(e, complexElementTy);
  } else
    real = CGF.buildScalarExpr(e);

  return createComplexFromReal(CGF.getBuilder(), CGF.getLoc(e->getExprLoc()),
                               real);
}

LValue ComplexExprEmitter::buildCompoundAssignLValue(
    const CompoundAssignOperator *e,
    mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &), RValue &val) {
  QualType lhsTy = e->getLHS()->getType();
  if (const AtomicType *at = lhsTy->getAs<AtomicType>())
    lhsTy = at->getValueType();

  BinOpInfo opInfo{CGF.getLoc(e->getExprLoc())};
  opInfo.fpFeatures = e->getFPFeaturesInEffect(CGF.getLangOpts());

  assert(!MissingFeatures::CGFPOptionsRAII());

  // Load the RHS and LHS operands.
  // __block variables need to have the rhs evaluated first, plus this should
  // improve codegen a little.
  QualType promotionTypeCr;
  promotionTypeCr = getPromotionType(e->getComputationResultType());
  if (promotionTypeCr.isNull())
    promotionTypeCr = e->getComputationResultType();
  opInfo.ty = promotionTypeCr;
  QualType complexElementTy =
      opInfo.ty->castAs<ComplexType>()->getElementType();
  QualType promotionTypeRhs = getPromotionType(e->getRHS()->getType());

  // The RHS should have been converted to the computation type.
  if (e->getRHS()->getType()->isRealFloatingType()) {
    if (!promotionTypeRhs.isNull())
      opInfo.rhs = createComplexFromReal(
          CGF.getBuilder(), CGF.getLoc(e->getExprLoc()),
          CGF.buildPromotedScalarExpr(e->getRHS(), promotionTypeRhs));
    else {
      assert(CGF.getContext().hasSameUnqualifiedType(complexElementTy,
                                                     e->getRHS()->getType()));
      opInfo.rhs =
          createComplexFromReal(CGF.getBuilder(), CGF.getLoc(e->getExprLoc()),
                                CGF.buildScalarExpr(e->getRHS()));
    }
  } else {
    if (!promotionTypeRhs.isNull()) {
      opInfo.rhs = createComplexFromReal(
          CGF.getBuilder(), CGF.getLoc(e->getExprLoc()),
          CGF.buildPromotedComplexExpr(e->getRHS(), promotionTypeRhs));
    } else {
      assert(CGF.getContext().hasSameUnqualifiedType(opInfo.ty,
                                                     e->getRHS()->getType()));
      opInfo.rhs = Visit(e->getRHS());
    }
  }

  LValue lhs = CGF.buildLValue(e->getLHS());

  // Load from the l-value and convert it.
  SourceLocation loc = e->getExprLoc();
  QualType promotionTypeLhs = getPromotionType(e->getComputationLHSType());
  if (lhsTy->isAnyComplexType()) {
    mlir::Value lhsVal = buildLoadOfLValue(lhs, loc);
    if (!promotionTypeLhs.isNull())
      opInfo.lhs =
          buildComplexToComplexCast(lhsVal, lhsTy, promotionTypeLhs, loc);
    else
      opInfo.lhs = buildComplexToComplexCast(lhsVal, lhsTy, opInfo.ty, loc);
  } else {
    mlir::Value lhsVal = CGF.buildLoadOfScalar(lhs, loc);
    // For floating point real operands we can directly pass the scalar form
    // to the binary operator emission and potentially get more efficient code.
    if (lhsTy->isRealFloatingType()) {
      QualType promotedComplexElementTy;
      if (!promotionTypeLhs.isNull()) {
        promotedComplexElementTy =
            cast<ComplexType>(promotionTypeLhs)->getElementType();
        if (!CGF.getContext().hasSameUnqualifiedType(promotedComplexElementTy,
                                                     promotionTypeLhs))
          lhsVal = CGF.buildScalarConversion(lhsVal, lhsTy,
                                             promotedComplexElementTy, loc);
      } else {
        if (!CGF.getContext().hasSameUnqualifiedType(complexElementTy, lhsTy))
          lhsVal =
              CGF.buildScalarConversion(lhsVal, lhsTy, complexElementTy, loc);
      }
      opInfo.lhs = createComplexFromReal(CGF.getBuilder(),
                                         CGF.getLoc(e->getExprLoc()), lhsVal);
    } else {
      opInfo.lhs = buildScalarToComplexCast(lhsVal, lhsTy, opInfo.ty, loc);
    }
  }

  // Expand the binary operator.
  mlir::Value result = (this->*func)(opInfo);

  // Truncate the result and store it into the LHS lvalue.
  if (lhsTy->isAnyComplexType()) {
    mlir::Value resVal =
        buildComplexToComplexCast(result, opInfo.ty, lhsTy, loc);
    buildStoreOfComplex(CGF.getLoc(e->getExprLoc()), resVal, lhs,
                        /*isInit*/ false);
    val = RValue::getComplex(resVal);
  } else {
    mlir::Value resVal =
        CGF.buildComplexToScalarConversion(result, opInfo.ty, lhsTy, loc);
    CGF.buildStoreOfScalar(resVal, lhs, /*isInit*/ false);
    val = RValue::get(resVal);
  }

  return lhs;
}

mlir::Value ComplexExprEmitter::buildCompoundAssign(
    const CompoundAssignOperator *e,
    mlir::Value (ComplexExprEmitter::*func)(const BinOpInfo &)) {
  RValue val;
  LValue lv = buildCompoundAssignLValue(e, func, val);

  // The result of an assignment in C is the assigned r-value.
  if (!CGF.getLangOpts().CPlusPlus)
    return val.getComplexVal();

  // If the lvalue is non-volatile, return the computed value of the assignment.
  if (!lv.isVolatileQualified())
    return val.getComplexVal();

  return buildLoadOfLValue(lv, e->getExprLoc());
}

mlir::Value ComplexExprEmitter::buildBinAdd(const BinOpInfo &op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexAdd(op.loc, op.lhs, op.rhs);
}

mlir::Value ComplexExprEmitter::buildBinSub(const BinOpInfo &op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexSub(op.loc, op.lhs, op.rhs);
}

static mlir::cir::ComplexRangeKind
getComplexRangeAttr(LangOptions::ComplexRangeKind range) {
  switch (range) {
  case LangOptions::CX_Full:
    return mlir::cir::ComplexRangeKind::Full;
  case LangOptions::CX_Improved:
    return mlir::cir::ComplexRangeKind::Improved;
  case LangOptions::CX_Promoted:
    return mlir::cir::ComplexRangeKind::Promoted;
  case LangOptions::CX_Basic:
    return mlir::cir::ComplexRangeKind::Basic;
  case LangOptions::CX_None:
    return mlir::cir::ComplexRangeKind::None;
  }
}

mlir::Value ComplexExprEmitter::buildBinMul(const BinOpInfo &op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexMul(
      op.loc, op.lhs, op.rhs,
      getComplexRangeAttr(op.fpFeatures.getComplexRange()), FPHasBeenPromoted);
}

mlir::Value ComplexExprEmitter::buildBinDiv(const BinOpInfo &op) {
  assert(!MissingFeatures::CGFPOptionsRAII());
  return CGF.getBuilder().createComplexDiv(
      op.loc, op.lhs, op.rhs,
      getComplexRangeAttr(op.fpFeatures.getComplexRange()), FPHasBeenPromoted);
}

LValue ComplexExprEmitter::buildBinAssignLValue(const BinaryOperator *e,
                                                mlir::Value &val) {
  assert(CGF.getContext().hasSameUnqualifiedType(e->getLHS()->getType(),
                                                 e->getRHS()->getType()) &&
         "Invalid assignment");

  // Emit the RHS.  __block variables need the RHS evaluated first.
  val = Visit(e->getRHS());

  // Compute the address to store into.
  LValue lhs = CGF.buildLValue(e->getLHS());

  // Store the result value into the LHS lvalue.
  buildStoreOfComplex(CGF.getLoc(e->getExprLoc()), val, lhs, /*isInit*/ false);

  return lhs;
}

mlir::Value
ComplexExprEmitter::VisitImaginaryLiteral(const ImaginaryLiteral *il) {
  auto loc = CGF.getLoc(il->getExprLoc());
  auto ty = mlir::cast<mlir::cir::ComplexType>(CGF.getCIRType(il->getType()));
  auto elementTy = ty.getElementTy();

  mlir::TypedAttr realValueAttr;
  mlir::TypedAttr imagValueAttr;
  if (mlir::isa<mlir::cir::IntType>(elementTy)) {
    auto imagValue = cast<IntegerLiteral>(il->getSubExpr())->getValue();
    realValueAttr = mlir::cir::IntAttr::get(elementTy, 0);
    imagValueAttr = mlir::cir::IntAttr::get(elementTy, imagValue);
  } else if (mlir::isa<mlir::cir::CIRFPTypeInterface>(elementTy)) {
    auto imagValue = cast<FloatingLiteral>(il->getSubExpr())->getValue();
    realValueAttr = mlir::cir::FPAttr::get(
        elementTy, llvm::APFloat::getZero(imagValue.getSemantics()));
    imagValueAttr = mlir::cir::FPAttr::get(elementTy, imagValue);
  } else
    llvm_unreachable("unexpected complex element type");

  auto realValue = builder.getConstant(loc, realValueAttr);
  auto imagValue = builder.getConstant(loc, imagValueAttr);
  return builder.createComplexCreate(loc, realValue, imagValue);
}

mlir::Value ComplexExprEmitter::VisitInitListExpr(InitListExpr *e) {
  if (e->getNumInits() == 2) {
    mlir::Value real = CGF.buildScalarExpr(e->getInit(0));
    mlir::Value imag = CGF.buildScalarExpr(e->getInit(1));
    return builder.createComplexCreate(CGF.getLoc(e->getExprLoc()), real, imag);
  }

  if (e->getNumInits() == 1)
    return Visit(e->getInit(0));

  // Empty init list initializes to null
  assert(e->getNumInits() == 0 && "Unexpected number of inits");
  QualType ty = e->getType()->castAs<ComplexType>()->getElementType();
  return builder.getZero(CGF.getLoc(e->getExprLoc()), CGF.ConvertType(ty));
}

mlir::Value CIRGenFunction::buildPromotedComplexExpr(const Expr *e,
                                                     QualType promotionType) {
  return ComplexExprEmitter(*this).buildPromoted(e, promotionType);
}

mlir::Value CIRGenFunction::buildPromotedValue(mlir::Value result,
                                               QualType promotionType) {
  assert(mlir::isa<mlir::cir::CIRFPTypeInterface>(
             mlir::cast<mlir::cir::ComplexType>(result.getType())
                 .getElementTy()) &&
         "integral complex will never be promoted");
  return builder.createCast(mlir::cir::CastKind::float_complex, result,
                            ConvertType(promotionType));
}

mlir::Value CIRGenFunction::buildUnPromotedValue(mlir::Value result,
                                                 QualType unPromotionType) {
  assert(mlir::isa<mlir::cir::CIRFPTypeInterface>(
             mlir::cast<mlir::cir::ComplexType>(result.getType())
                 .getElementTy()) &&
         "integral complex will never be promoted");
  return builder.createCast(mlir::cir::CastKind::float_complex, result,
                            ConvertType(unPromotionType));
}

mlir::Value CIRGenFunction::buildComplexExpr(const Expr *e) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");

  return ComplexExprEmitter(*this).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::buildComplexExprIntoLValue(const Expr *e, LValue dest,
                                                bool isInit) {
  assert(e && getComplexType(e->getType()) &&
         "Invalid complex expression to emit");
  ComplexExprEmitter emitter(*this);
  mlir::Value val = emitter.Visit(const_cast<Expr *>(e));
  emitter.buildStoreOfComplex(getLoc(e->getExprLoc()), val, dest, isInit);
}

void CIRGenFunction::buildStoreOfComplex(mlir::Location loc, mlir::Value v,
                                         LValue dest, bool isInit) {
  ComplexExprEmitter(*this).buildStoreOfComplex(loc, v, dest, isInit);
}

Address CIRGenFunction::buildAddrOfRealComponent(mlir::Location loc,
                                                 Address addr,
                                                 QualType complexType) {
  return builder.createRealPtr(loc, addr);
}

Address CIRGenFunction::buildAddrOfImagComponent(mlir::Location loc,
                                                 Address addr,
                                                 QualType complexType) {
  return builder.createImagPtr(loc, addr);
}

LValue CIRGenFunction::buildComplexAssignmentLValue(const BinaryOperator *e) {
  assert(e->getOpcode() == BO_Assign);
  mlir::Value val; // ignored
  LValue lVal = ComplexExprEmitter(*this).buildBinAssignLValue(e, val);
  if (getLangOpts().OpenMP)
    llvm_unreachable("NYI");
  return lVal;
}

using CompoundFunc =
    mlir::Value (ComplexExprEmitter::*)(const ComplexExprEmitter::BinOpInfo &);

static CompoundFunc getComplexOp(BinaryOperatorKind op) {
  switch (op) {
  case BO_MulAssign:
    return &ComplexExprEmitter::buildBinMul;
  case BO_DivAssign:
    return &ComplexExprEmitter::buildBinDiv;
  case BO_SubAssign:
    return &ComplexExprEmitter::buildBinSub;
  case BO_AddAssign:
    return &ComplexExprEmitter::buildBinAdd;
  default:
    llvm_unreachable("unexpected complex compound assignment");
  }
}

LValue CIRGenFunction::buildComplexCompoundAssignmentLValue(
    const CompoundAssignOperator *e) {
  CompoundFunc op = getComplexOp(e->getOpcode());
  RValue val;
  return ComplexExprEmitter(*this).buildCompoundAssignLValue(e, op, val);
}

mlir::Value CIRGenFunction::buildComplexPrePostIncDec(const UnaryOperator *e,
                                                      LValue lv, bool isInc,
                                                      bool isPre) {
  mlir::Value inVal = buildLoadOfComplex(lv, e->getExprLoc());

  auto loc = getLoc(e->getExprLoc());
  auto opKind =
      isInc ? mlir::cir::UnaryOpKind::Inc : mlir::cir::UnaryOpKind::Dec;
  mlir::Value incVal = builder.createUnaryOp(loc, opKind, inVal);

  // Store the updated result through the lvalue.
  buildStoreOfComplex(loc, incVal, lv, /*init*/ false);
  if (getLangOpts().OpenMP)
    llvm_unreachable("NYI");

  // If this is a postinc, return the value read from memory, otherwise use the
  // updated value.
  return isPre ? incVal : inVal;
}

mlir::Value CIRGenFunction::buildLoadOfComplex(LValue src, SourceLocation loc) {
  return ComplexExprEmitter(*this).buildLoadOfLValue(src, loc);
}
