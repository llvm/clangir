//===--- CIRGenExprAgg.cpp - Emit CIR Code from Aggregate Expressions -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Aggregate Expr nodes as CIR code.
//
//===----------------------------------------------------------------------===//
#include "CIRGenCall.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "CIRGenTypes.h"
#include "CIRGenValue.h"
#include "mlir/IR/Attributes.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace cir;
using namespace clang;

namespace {

// FIXME(cir): This should be a common helper between CIRGen
// and traditional CodeGen
/// Is the value of the given expression possibly a reference to or
/// into a __block variable?
static bool isBlockVarRef(const Expr *E) {
  // Make sure we look through parens.
  E = E->IgnoreParens();

  // Check for a direct reference to a __block variable.
  if (const DeclRefExpr *dre = dyn_cast<DeclRefExpr>(E)) {
    const VarDecl *var = dyn_cast<VarDecl>(dre->getDecl());
    return (var && var->hasAttr<BlocksAttr>());
  }

  // More complicated stuff.

  // Binary operators.
  if (const BinaryOperator *op = dyn_cast<BinaryOperator>(E)) {
    // For an assignment or pointer-to-member operation, just care
    // about the LHS.
    if (op->isAssignmentOp() || op->isPtrMemOp())
      return isBlockVarRef(op->getLHS());

    // For a comma, just care about the RHS.
    if (op->getOpcode() == BO_Comma)
      return isBlockVarRef(op->getRHS());

    // FIXME: pointer arithmetic?
    return false;

    // Check both sides of a conditional operator.
  }
  if (const AbstractConditionalOperator *op =
          dyn_cast<AbstractConditionalOperator>(E)) {
    return isBlockVarRef(op->getTrueExpr()) ||
           isBlockVarRef(op->getFalseExpr());

    // OVEs are required to support BinaryConditionalOperators.
  } else if (const OpaqueValueExpr *op = dyn_cast<OpaqueValueExpr>(E)) {
    if (const Expr *src = op->getSourceExpr())
      return isBlockVarRef(src);

    // Casts are necessary to get things like (*(int*)&var) = foo().
    // We don't really care about the kind of cast here, except
    // we don't want to look through l2r casts, because it's okay
    // to get the *value* in a __block variable.
  } else if (const CastExpr *cast = dyn_cast<CastExpr>(E)) {
    if (cast->getCastKind() == CK_LValueToRValue)
      return false;
    return isBlockVarRef(cast->getSubExpr());

    // Handle unary operators.  Again, just aggressively look through
    // it, ignoring the operation.
  } else if (const UnaryOperator *uop = dyn_cast<UnaryOperator>(E)) {
    return isBlockVarRef(uop->getSubExpr());

    // Look into the base of a field access.
  } else if (const MemberExpr *mem = dyn_cast<MemberExpr>(E)) {
    return isBlockVarRef(mem->getBase());

    // Look into the base of a subscript.
  } else if (const ArraySubscriptExpr *sub = dyn_cast<ArraySubscriptExpr>(E)) {
    return isBlockVarRef(sub->getBase());
  }

  return false;
}

class AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CIRGenFunction &CGF;
  AggValueSlot dest;
  bool isResultUnused;

  // Calls `Fn` with a valid return value slot, potentially creating a temporary
  // to do so. If a temporary is created, an appropriate copy into `Dest` will
  // be emitted, as will lifetime markers.
  //
  // The given function should take a ReturnValueSlot, and return an RValue that
  // points to said slot.
  void withReturnValueSlot(const Expr *e,
                           llvm::function_ref<RValue(ReturnValueSlot)> fn);

  AggValueSlot ensureSlot(mlir::Location loc, QualType t) {
    if (!dest.isIgnored())
      return dest;
    return CGF.CreateAggTemp(t, loc, "agg.tmp.ensured");
  }

  void ensureDest(mlir::Location loc, QualType t) {
    if (!dest.isIgnored())
      return;
    dest = CGF.CreateAggTemp(t, loc, "agg.tmp.ensured");
  }

public:
  AggExprEmitter(CIRGenFunction &cgf, AggValueSlot dest, bool isResultUnused)
      : CGF{cgf}, dest(dest), isResultUnused(isResultUnused) {}

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// Given an expression with aggregate type that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result into
  /// DestPtr.
  void buildAggLoadOfLValue(const Expr *e);

  enum ExprValueKind { EVK_RValue, EVK_NonRValue };

  /// Perform the final copy to DestPtr, if desired.
  void buildFinalDestCopy(QualType type, RValue src);

  /// Perform the final copy to DestPtr, if desired. SrcIsRValue is true if
  /// source comes from an RValue.
  void buildFinalDestCopy(QualType type, const LValue &src,
                          ExprValueKind srcValueKind = EVK_NonRValue);
  void buildCopy(QualType type, const AggValueSlot &dest,
                 const AggValueSlot &src);

  void buildArrayInit(Address destPtr, mlir::cir::ArrayType aType,
                      QualType arrayQTy, Expr *exprToVisit,
                      ArrayRef<Expr *> args, Expr *arrayFiller);

  AggValueSlot::NeedsGCBarriers_t needsGC(QualType t) {
    if (CGF.getLangOpts().getGC() && typeRequiresGCollection(t))
      llvm_unreachable("garbage collection is NYI");
    return AggValueSlot::DoesNotNeedGCBarriers;
  }

  bool typeRequiresGCollection(QualType t);

  //===--------------------------------------------------------------------===//
  //                             Visitor Methods
  //===--------------------------------------------------------------------===//

  void Visit(Expr *e) {
    if (CGF.getDebugInfo()) {
      llvm_unreachable("NYI");
    }
    StmtVisitor<AggExprEmitter>::Visit(e);
  }

  void VisitStmt(Stmt *s) {
    llvm::errs() << "Missing visitor for AggExprEmitter Stmt: "
                 << s->getStmtClassName() << "\n";
    llvm_unreachable("NYI");
  }
  void VisitParenExpr(ParenExpr *pe) { Visit(pe->getSubExpr()); }
  void VisitGenericSelectionExpr(GenericSelectionExpr *ge) {
    llvm_unreachable("NYI");
  }
  void VisitCoawaitExpr(CoawaitExpr *e) {
    CGF.buildCoawaitExpr(*e, dest, isResultUnused);
  }
  void VisitCoyieldExpr(CoyieldExpr *e) { llvm_unreachable("NYI"); }
  void VisitUnaryCoawait(UnaryOperator *e) { llvm_unreachable("NYI"); }
  void VisitUnaryExtension(UnaryOperator *e) { llvm_unreachable("NYI"); }
  void VisitSubstNonTypeTemplateParmExpr(SubstNonTypeTemplateParmExpr *e) {
    llvm_unreachable("NYI");
  }
  void VisitConstantExpr(ConstantExpr *e) { llvm_unreachable("NYI"); }

  // l-values
  void VisitDeclRefExpr(DeclRefExpr *e) { buildAggLoadOfLValue(e); }
  void VisitMemberExpr(MemberExpr *e) { buildAggLoadOfLValue(e); }
  void VisitUnaryDeref(UnaryOperator *e) { buildAggLoadOfLValue(e); }
  void VisitStringLiteral(StringLiteral *e) { llvm_unreachable("NYI"); }
  void VisitCompoundLiteralExpr(CompoundLiteralExpr *e);
  void VisitArraySubscriptExpr(ArraySubscriptExpr *e) {
    buildAggLoadOfLValue(e);
  }
  void VisitPredefinedExpr(const PredefinedExpr *e) { llvm_unreachable("NYI"); }

  // Operators.
  void VisitCastExpr(CastExpr *e);
  void VisitCallExpr(const CallExpr *e);

  void VisitStmtExpr(const StmtExpr *e) {
    assert(!MissingFeatures::stmtExprEvaluation() && "NYI");
    CGF.buildCompoundStmt(*e->getSubStmt(), /*getLast=*/true, dest);
  }

  void VisitBinaryOperator(const BinaryOperator *e) { llvm_unreachable("NYI"); }
  void visitPointerToDataMemberBinaryOperator(const BinaryOperator *e) {
    llvm_unreachable("NYI");
  }
  void VisitBinAssign(const BinaryOperator *e) {

    // For an assignment to work, the value on the right has
    // to be compatible with the value on the left.
    assert(CGF.getContext().hasSameUnqualifiedType(e->getLHS()->getType(),
                                                   e->getRHS()->getType()) &&
           "Invalid assignment");

    if (isBlockVarRef(e->getLHS()) &&
        e->getRHS()->HasSideEffects(CGF.getContext())) {
      llvm_unreachable("NYI");
    }

    LValue lhs = CGF.buildLValue(e->getLHS());

    // If we have an atomic type, evaluate into the destination and then
    // do an atomic copy.
    if (lhs.getType()->isAtomicType() ||
        CGF.LValueIsSuitableForInlineAtomic(lhs)) {
      assert(!MissingFeatures::atomicTypes());
      return;
    }

    // Codegen the RHS so that it stores directly into the LHS.
    AggValueSlot lhsSlot = AggValueSlot::forLValue(
        lhs, AggValueSlot::IsDestructed, AggValueSlot::DoesNotNeedGCBarriers,
        AggValueSlot::IsAliased, AggValueSlot::MayOverlap);

    // A non-volatile aggregate destination might have volatile member.
    if (!lhsSlot.isVolatile() && CGF.hasVolatileMember(e->getLHS()->getType()))
      assert(!MissingFeatures::atomicTypes());

    CGF.buildAggExpr(e->getRHS(), lhsSlot);

    // Copy into the destination if the assignment isn't ignored.
    buildFinalDestCopy(e->getType(), lhs);

    if (!dest.isIgnored() && !dest.isExternallyDestructed() &&
        e->getType().isDestructedType() == QualType::DK_nontrivial_c_struct)
      CGF.pushDestroy(QualType::DK_nontrivial_c_struct, dest.getAddress(),
                      e->getType());
  }

  void VisitBinComma(const BinaryOperator *e);
  void VisitBinCmp(const BinaryOperator *e);
  void VisitCXXRewrittenBinaryOperator(CXXRewrittenBinaryOperator *e) {
    llvm_unreachable("NYI");
  }

  void VisitObjCMessageExpr(ObjCMessageExpr *e) { llvm_unreachable("NYI"); }
  void visitObjCiVarRefExpr(ObjCIvarRefExpr *e) { llvm_unreachable("NYI"); }

  void VisitDesignatedInitUpdateExpr(DesignatedInitUpdateExpr *e) {
    llvm_unreachable("NYI");
  }
  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *e);
  void VisitChooseExpr(const ChooseExpr *e) { llvm_unreachable("NYI"); }
  void VisitInitListExpr(InitListExpr *e);
  void VisitCXXParenListInitExpr(CXXParenListInitExpr *e);
  void visitCxxParenListOrInitListExpr(Expr *exprToVisit, ArrayRef<Expr *> args,
                                       FieldDecl *initializedFieldInUnion,
                                       Expr *arrayFiller);
  void VisitArrayInitLoopExpr(const ArrayInitLoopExpr *e,
                              llvm::Value *outerBegin = nullptr) {
    llvm_unreachable("NYI");
  }
  void VisitImplicitValueInitExpr(ImplicitValueInitExpr *e) {
    llvm_unreachable("NYI");
  }
  void VisitNoInitExpr(NoInitExpr *e) { llvm_unreachable("NYI"); }
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *dae) {
    CIRGenFunction::CXXDefaultArgExprScope scope(CGF, dae);
    Visit(dae->getExpr());
  }
  void VisitCXXDefaultInitExpr(CXXDefaultInitExpr *die) {
    CIRGenFunction::CXXDefaultInitExprScope scope(CGF, die);
    Visit(die->getExpr());
  }
  void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *e);
  void VisitCXXConstructExpr(const CXXConstructExpr *e);
  void VisitCXXInheritedCtorInitExpr(const CXXInheritedCtorInitExpr *e) {
    llvm_unreachable("NYI");
  }
  void VisitLambdaExpr(LambdaExpr *e);
  void VisitCXXStdInitializerListExpr(CXXStdInitializerListExpr *e) {
    ASTContext &ctx = CGF.getContext();
    CIRGenFunction::SourceLocRAIIObject locRAIIObject{
        CGF, CGF.getLoc(e->getSourceRange())};
    // Emit an array containing the elements.  The array is externally
    // destructed if the std::initializer_list object is.
    LValue array = CGF.buildLValue(e->getSubExpr());
    assert(array.isSimple() && "initializer_list array not a simple lvalue");
    Address arrayPtr = array.getAddress();

    const ConstantArrayType *arrayType =
        ctx.getAsConstantArrayType(e->getSubExpr()->getType());
    assert(arrayType && "std::initializer_list constructed from non-array");

    RecordDecl *record = e->getType()->castAs<RecordType>()->getDecl();
    RecordDecl::field_iterator field = record->field_begin();
    assert(field != record->field_end() &&
           ctx.hasSameType(field->getType()->getPointeeType(),
                           arrayType->getElementType()) &&
           "Expected std::initializer_list first field to be const E *");
    // Start pointer.
    auto loc = CGF.getLoc(e->getSourceRange());
    AggValueSlot dest = ensureSlot(loc, e->getType());
    LValue destLv = CGF.makeAddrLValue(dest.getAddress(), e->getType());
    LValue start =
        CGF.buildLValueForFieldInitialization(destLv, *field, field->getName());
    mlir::Value arrayStart = arrayPtr.emitRawPointer();
    CGF.buildStoreThroughLValue(RValue::get(arrayStart), start);
    ++field;
    assert(field != record->field_end() &&
           "Expected std::initializer_list to have two fields");

    auto builder = CGF.getBuilder();

    auto sizeOp = builder.getConstInt(loc, arrayType->getSize());

    mlir::Value size = sizeOp.getRes();
    builder.getUIntNTy(arrayType->getSizeBitWidth());
    LValue endOrLength =
        CGF.buildLValueForFieldInitialization(destLv, *field, field->getName());
    if (ctx.hasSameType(field->getType(), ctx.getSizeType())) {
      // Length.
      CGF.buildStoreThroughLValue(RValue::get(size), endOrLength);
    } else {
      // End pointer.
      assert(field->getType()->isPointerType() &&
             ctx.hasSameType(field->getType()->getPointeeType(),
                             arrayType->getElementType()) &&
             "Expected std::initializer_list second field to be const E *");

      auto arrayEnd =
          builder.getArrayElement(loc, loc, arrayPtr.getPointer(),
                                  arrayPtr.getElementType(), size, false);
      CGF.buildStoreThroughLValue(RValue::get(arrayEnd), endOrLength);
    }
    assert(++field == record->field_end() &&
           "Expected std::initializer_list to only have two fields");
  }

  void VisitExprWithCleanups(ExprWithCleanups *e);
  void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *e) {
    llvm_unreachable("NYI");
  }
  void VisitCXXTypeidExpr(CXXTypeidExpr *e) { llvm_unreachable("NYI"); }
  void VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *e);
  void VisitOpaqueValueExpr(OpaqueValueExpr *e) { llvm_unreachable("NYI"); }

  void VisitPseudoObjectExpr(PseudoObjectExpr *e) { llvm_unreachable("NYI"); }

  void VisitVAArgExpr(VAArgExpr *e) { llvm_unreachable("NYI"); }

  void buildInitializationToLValue(Expr *e, LValue lv);

  void buildNullInitializationToLValue(mlir::Location loc, LValue address);
  void VisitCXXThrowExpr(const CXXThrowExpr *e) { llvm_unreachable("NYI"); }
  void VisitAtomicExpr(AtomicExpr *e) { llvm_unreachable("NYI"); }
};
} // namespace

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// Given an expression with aggregate type that represents a value lvalue, this
/// method emits the address of the lvalue, then loads the result into DestPtr.
void AggExprEmitter::buildAggLoadOfLValue(const Expr *e) {
  LValue lv = CGF.buildLValue(e);

  // If the type of the l-value is atomic, then do an atomic load.
  if (lv.getType()->isAtomicType() || CGF.LValueIsSuitableForInlineAtomic(lv) ||
      MissingFeatures::atomicTypes())
    llvm_unreachable("atomic load is NYI");

  buildFinalDestCopy(e->getType(), lv);
}

/// Perform the final copy to DestPtr, if desired.
void AggExprEmitter::buildFinalDestCopy(QualType type, RValue src) {
  assert(src.isAggregate() && "value must be aggregate value!");
  LValue srcLV = CGF.makeAddrLValue(src.getAggregateAddress(), type);
  buildFinalDestCopy(type, srcLV, EVK_RValue);
}

/// Perform the final copy to DestPtr, if desired.
void AggExprEmitter::buildFinalDestCopy(QualType type, const LValue &src,
                                        ExprValueKind srcValueKind) {
  // If Dest is ignored, then we're evaluating an aggregate expression
  // in a context that doesn't care about the result.  Note that loads
  // from volatile l-values force the existence of a non-ignored
  // destination.
  if (dest.isIgnored())
    return;

  // Copy non-trivial C structs here.
  if (dest.isVolatile())
    assert(!MissingFeatures::volatileTypes());

  if (srcValueKind == EVK_RValue) {
    if (type.isNonTrivialToPrimitiveDestructiveMove() == QualType::PCK_Struct) {
      llvm_unreachable("move assignment/move ctor for rvalue is NYI");
    }
  } else {
    if (type.isNonTrivialToPrimitiveCopy() == QualType::PCK_Struct)
      llvm_unreachable("non-trivial primitive copy is NYI");
  }

  AggValueSlot srcAgg = AggValueSlot::forLValue(
      src, AggValueSlot::IsDestructed, needsGC(type), AggValueSlot::IsAliased,
      AggValueSlot::MayOverlap);
  buildCopy(type, dest, srcAgg);
}

/// Perform a copy from the source into the destination.
///
/// \param type - the type of the aggregate being copied; qualifiers are
///   ignored
void AggExprEmitter::buildCopy(QualType type, const AggValueSlot &dest,
                               const AggValueSlot &src) {
  if (dest.requiresGCollection())
    llvm_unreachable("garbage collection is NYI");

  // If the result of the assignment is used, copy the LHS there also.
  // It's volatile if either side is.  Use the minimum alignment of
  // the two sides.
  LValue destLv = CGF.makeAddrLValue(dest.getAddress(), type);
  LValue srcLv = CGF.makeAddrLValue(src.getAddress(), type);
  CGF.buildAggregateCopy(destLv, srcLv, type, dest.mayOverlap(),
                         dest.isVolatile() || src.isVolatile());
}

// FIXME(cir): This function could be shared with traditional LLVM codegen
/// Determine if E is a trivial array filler, that is, one that is
/// equivalent to zero-initialization.
static bool isTrivialFiller(Expr *e) {
  if (!e)
    return true;

  if (isa<ImplicitValueInitExpr>(e))
    return true;

  if (auto *ile = dyn_cast<InitListExpr>(e)) {
    if (ile->getNumInits())
      return false;
    return isTrivialFiller(ile->getArrayFiller());
  }

  if (auto *cons = dyn_cast_or_null<CXXConstructExpr>(e))
    return cons->getConstructor()->isDefaultConstructor() &&
           cons->getConstructor()->isTrivial();

  // FIXME: Are there other cases where we can avoid emitting an initializer?
  return false;
}

void AggExprEmitter::buildArrayInit(Address destPtr, mlir::cir::ArrayType aType,
                                    QualType arrayQTy, Expr *exprToVisit,
                                    ArrayRef<Expr *> args, Expr *arrayFiller) {
  uint64_t numInitElements = args.size();

  uint64_t numArrayElements = aType.getSize();
  assert(numInitElements <= numArrayElements);

  QualType elementType =
      CGF.getContext().getAsArrayType(arrayQTy)->getElementType();
  QualType elementPtrType = CGF.getContext().getPointerType(elementType);

  auto cirElementType = CGF.convertType(elementType);
  auto cirAddrSpace = mlir::cast_if_present<mlir::cir::AddressSpaceAttr>(
      destPtr.getType().getAddrSpace());
  auto cirElementPtrType =
      CGF.getBuilder().getPointerTo(cirElementType, cirAddrSpace);
  auto loc = CGF.getLoc(exprToVisit->getSourceRange());

  // Cast from cir.ptr<cir.array<elementType> to cir.ptr<elementType>
  auto begin = CGF.getBuilder().create<mlir::cir::CastOp>(
      loc, cirElementPtrType, mlir::cir::CastKind::array_to_ptrdecay,
      destPtr.getPointer());

  CharUnits elementSize = CGF.getContext().getTypeSizeInChars(elementType);
  CharUnits elementAlign =
      destPtr.getAlignment().alignmentOfArrayElement(elementSize);

  // Exception safety requires us to destroy all the
  // already-constructed members if an initializer throws.
  // For that, we'll need an EH cleanup.
  QualType::DestructionKind dtorKind = elementType.isDestructedType();
  [[maybe_unused]] Address endOfInit = Address::invalid();
  CIRGenFunction::CleanupDeactivationScope deactivation(CGF);

  if (dtorKind) {
    llvm_unreachable("dtorKind NYI");
  }

  // The 'current element to initialize'.  The invariants on this
  // variable are complicated.  Essentially, after each iteration of
  // the loop, it points to the last initialized element, except
  // that it points to the beginning of the array before any
  // elements have been initialized.
  mlir::Value element = begin;

  // Don't build the 'one' before the cycle to avoid
  // emmiting the redundant `cir.const 1` instrs.
  mlir::Value one;

  // Emit the explicit initializers.
  for (uint64_t i = 0; i != numInitElements; ++i) {
    if (i == 1)
      one = CGF.getBuilder().getConstInt(
          loc, mlir::cast<mlir::cir::IntType>(CGF.PtrDiffTy), 1);

    // Advance to the next element.
    if (i > 0) {
      element = CGF.getBuilder().create<mlir::cir::PtrStrideOp>(
          loc, cirElementPtrType, element, one);

      // Tell the cleanup that it needs to destroy up to this
      // element.  TODO: some of these stores can be trivially
      // observed to be unnecessary.
      assert(!endOfInit.isValid() && "destructed types NIY");
    }

    LValue elementLV = CGF.makeAddrLValue(
        Address(element, cirElementType, elementAlign), elementType);
    buildInitializationToLValue(args[i], elementLV);
  }

  // Check whether there's a non-trivial array-fill expression.
  bool hasTrivialFiller = isTrivialFiller(arrayFiller);

  // Any remaining elements need to be zero-initialized, possibly
  // using the filler expression.  We can skip this if the we're
  // emitting to zeroed memory.
  if (numInitElements != numArrayElements &&
      !(dest.isZeroed() && hasTrivialFiller &&
        CGF.getTypes().isZeroInitializable(elementType))) {

    // Use an actual loop.  This is basically
    //   do { *array++ = filler; } while (array != end);

    auto &builder = CGF.getBuilder();

    // Advance to the start of the rest of the array.
    if (numInitElements) {
      auto one = builder.getConstInt(
          loc, mlir::cast<mlir::cir::IntType>(CGF.PtrDiffTy), 1);
      element = builder.create<mlir::cir::PtrStrideOp>(loc, cirElementPtrType,
                                                       element, one);

      assert(!endOfInit.isValid() && "destructed types NIY");
    }

    // Allocate the temporary variable
    // to store the pointer to first unitialized element
    auto tmpAddr = CGF.CreateTempAlloca(
        cirElementPtrType, CGF.getPointerAlign(), loc, "arrayinit.temp");
    LValue tmpLV = CGF.makeAddrLValue(tmpAddr, elementPtrType);
    CGF.buildStoreThroughLValue(RValue::get(element), tmpLV);

    // Compute the end of array
    auto numArrayElementsConst = builder.getConstInt(
        loc, mlir::cast<mlir::cir::IntType>(CGF.PtrDiffTy), numArrayElements);
    mlir::Value end = builder.create<mlir::cir::PtrStrideOp>(
        loc, cirElementPtrType, begin, numArrayElementsConst);

    builder.createDoWhile(
        loc,
        /*condBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto currentElement = builder.createLoad(loc, tmpAddr);
          mlir::Type boolTy = CGF.getCIRType(CGF.getContext().BoolTy);
          auto cmp = builder.create<mlir::cir::CmpOp>(
              loc, boolTy, mlir::cir::CmpOpKind::ne, currentElement, end);
          builder.createCondition(cmp);
        },
        /*bodyBuilder=*/
        [&](mlir::OpBuilder &b, mlir::Location loc) {
          auto currentElement = builder.createLoad(loc, tmpAddr);

          if (MissingFeatures::cleanups())
            llvm_unreachable("NYI");

          // Emit the actual filler expression.
          LValue elementLV = CGF.makeAddrLValue(
              Address(currentElement, cirElementType, elementAlign),
              elementType);
          if (arrayFiller)
            buildInitializationToLValue(arrayFiller, elementLV);
          else
            buildNullInitializationToLValue(loc, elementLV);

          // Tell the EH cleanup that we finished with the last element.
          assert(!endOfInit.isValid() && "destructed types NIY");

          // Advance pointer and store them to temporary variable
          auto one = builder.getConstInt(
              loc, mlir::cast<mlir::cir::IntType>(CGF.PtrDiffTy), 1);
          auto nextElement = builder.create<mlir::cir::PtrStrideOp>(
              loc, cirElementPtrType, currentElement, one);
          CGF.buildStoreThroughLValue(RValue::get(nextElement), tmpLV);

          builder.createYield(loc);
        });
  }
}

/// True if the given aggregate type requires special GC API calls.
bool AggExprEmitter::typeRequiresGCollection(QualType t) {
  // Only record types have members that might require garbage collection.
  const RecordType *recordTy = t->getAs<RecordType>();
  if (!recordTy)
    return false;

  // Don't mess with non-trivial C++ types.
  RecordDecl *record = recordTy->getDecl();
  if (isa<CXXRecordDecl>(record) &&
      (cast<CXXRecordDecl>(record)->hasNonTrivialCopyConstructor() ||
       !cast<CXXRecordDecl>(record)->hasTrivialDestructor()))
    return false;

  // Check whether the type has an object member.
  return record->hasObjectMember();
}

//===----------------------------------------------------------------------===//
//                             Visitor Methods
//===----------------------------------------------------------------------===//

/// Determine whether the given cast kind is known to always convert values
/// with all zero bits in their value representation to values with all zero
/// bits in their value representation.
/// TODO(cir): this can be shared with LLVM codegen.
static bool castPreservesZero(const CastExpr *ce) {
  switch (ce->getCastKind()) {
  case CK_HLSLVectorTruncation:
  case CK_HLSLArrayRValue:
    llvm_unreachable("NYI");
    // No-ops.
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_ConstructorConversion:
  case CK_BitCast:
  case CK_ToUnion:
  case CK_ToVoid:
    // Conversions between (possibly-complex) integral, (possibly-complex)
    // floating-point, and bool.
  case CK_BooleanToSignedIntegral:
  case CK_FloatingCast:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexToIntegralComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingRealToComplex:
  case CK_FloatingToBoolean:
  case CK_FloatingToIntegral:
  case CK_IntegralCast:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexToFloatingComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralRealToComplex:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
    // Reinterpreting integers as pointers and vice versa.
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
    // Language extensions.
  case CK_VectorSplat:
  case CK_MatrixCast:
  case CK_NonAtomicToAtomic:
  case CK_AtomicToNonAtomic:
    return true;

  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_NullToMemberPointer:
  case CK_ReinterpretMemberPointer:
    // FIXME: ABI-dependent.
    return false;

  case CK_AnyPointerToBlockPointerCast:
  case CK_BlockPointerToObjCPointerCast:
  case CK_CPointerToObjCPointerCast:
  case CK_ObjCObjectLValueCast:
  case CK_IntToOCLSampler:
  case CK_ZeroToOCLOpaqueType:
    // FIXME: Check these.
    return false;

  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToFloating:
  case CK_FixedPointToIntegral:
  case CK_FloatingToFixedPoint:
  case CK_IntegralToFixedPoint:
    // FIXME: Do all fixed-point types represent zero as all 0 bits?
    return false;

  case CK_AddressSpaceConversion:
  case CK_BaseToDerived:
  case CK_DerivedToBase:
  case CK_Dynamic:
  case CK_NullToPointer:
  case CK_PointerToBoolean:
    // FIXME: Preserves zeroes only if zero pointers and null pointers have the
    // same representation in all involved address spaces.
    return false;

  case CK_ARCConsumeObject:
  case CK_ARCExtendBlockObject:
  case CK_ARCProduceObject:
  case CK_ARCReclaimReturnedObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_BuiltinFnToFnPtr:
  case CK_Dependent:
  case CK_LValueBitCast:
  case CK_LValueToRValue:
  case CK_LValueToRValueBitCast:
  case CK_UncheckedDerivedToBase:
    return false;
  }
  llvm_unreachable("Unhandled clang::CastKind enum");
}

/// If emitting this value will obviously just cause a store of
/// zero to memory, return true.  This can return false if uncertain, so it just
/// handles simple cases.
static bool isSimpleZero(const Expr *e, CIRGenFunction &cgf) {
  e = e->IgnoreParens();
  while (auto *ce = dyn_cast<CastExpr>(e)) {
    if (!castPreservesZero(ce))
      break;
    e = ce->getSubExpr()->IgnoreParens();
  }

  // 0
  if (const IntegerLiteral *il = dyn_cast<IntegerLiteral>(e))
    return il->getValue() == 0;
  // +0.0
  if (const FloatingLiteral *fl = dyn_cast<FloatingLiteral>(e))
    return fl->getValue().isPosZero();
  // int()
  if ((isa<ImplicitValueInitExpr>(e) || isa<CXXScalarValueInitExpr>(e)) &&
      cgf.getTypes().isZeroInitializable(e->getType()))
    return true;
  // (int*)0 - Null pointer expressions.
  if (const CastExpr *ice = dyn_cast<CastExpr>(e)) {
    return ice->getCastKind() == CK_NullToPointer &&
           cgf.getTypes().isPointerZeroInitializable(e->getType()) &&
           !e->HasSideEffects(cgf.getContext());
  }
  // '\0'
  if (const CharacterLiteral *cl = dyn_cast<CharacterLiteral>(e))
    return cl->getValue() == 0;

  // Otherwise, hard case: conservatively return false.
  return false;
}

void AggExprEmitter::buildNullInitializationToLValue(mlir::Location loc,
                                                     LValue lv) {
  QualType type = lv.getType();

  // If the destination slot is already zeroed out before the aggregate is
  // copied into it, we don't have to emit any zeros here.
  if (dest.isZeroed() && CGF.getTypes().isZeroInitializable(type))
    return;

  if (CGF.hasScalarEvaluationKind(type)) {
    // For non-aggregates, we can store the appropriate null constant.
    auto null = CGF.cgm.buildNullConstant(type, loc);
    // Note that the following is not equivalent to
    // EmitStoreThroughBitfieldLValue for ARC types.
    if (lv.isBitField()) {
      mlir::Value result;
      CGF.buildStoreThroughBitfieldLValue(RValue::get(null), lv, result);
    } else {
      assert(lv.isSimple());
      CGF.buildStoreOfScalar(null, lv, /* isInitialization */ true);
    }
  } else {
    // There's a potential optimization opportunity in combining
    // memsets; that would be easy for arrays, but relatively
    // difficult for structures with the current code.
    CGF.buildNullInitialization(loc, lv.getAddress(), lv.getType());
  }
}

void AggExprEmitter::buildInitializationToLValue(Expr *E, LValue LV) {
  QualType type = LV.getType();
  // FIXME: Ignore result?
  // FIXME: Are initializers affected by volatile?
  if (dest.isZeroed() && isSimpleZero(E, CGF)) {
    // TODO(cir): LLVM codegen considers 'storing "i32 0" to a zero'd memory
    // location is a noop'. Consider emitting the store to zero in CIR, as to
    // model the actual user behavior, we can have a pass to optimize this out
    // later.
    return;
  }

  if (isa<ImplicitValueInitExpr>(E) || isa<CXXScalarValueInitExpr>(E)) {
    auto loc = E->getSourceRange().isValid() ? CGF.getLoc(E->getSourceRange())
                                             : *CGF.currSrcLoc;
    return buildNullInitializationToLValue(loc, LV);
  }
  if (isa<NoInitExpr>(E)) {
    // Do nothing.
    return;
  } else if (type->isReferenceType()) {
    RValue RV = CGF.buildReferenceBindingToExpr(E);
    return CGF.buildStoreThroughLValue(RV, LV);
  }

  switch (CGF.getEvaluationKind(type)) {
  case TEK_Complex:
    llvm_unreachable("NYI");
    return;
  case TEK_Aggregate:
    CGF.buildAggExpr(
        E, AggValueSlot::forLValue(LV, AggValueSlot::IsDestructed,
                                   AggValueSlot::DoesNotNeedGCBarriers,
                                   AggValueSlot::IsNotAliased,
                                   AggValueSlot::MayOverlap, dest.isZeroed()));
    return;
  case TEK_Scalar:
    if (LV.isSimple()) {
      CGF.buildScalarInit(E, CGF.getLoc(E->getSourceRange()), LV);
    } else {
      CGF.buildStoreThroughLValue(RValue::get(CGF.buildScalarExpr(E)), LV);
    }
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

void AggExprEmitter::VisitMaterializeTemporaryExpr(
    MaterializeTemporaryExpr *e) {
  Visit(e->getSubExpr());
}

void AggExprEmitter::VisitCXXConstructExpr(const CXXConstructExpr *e) {
  AggValueSlot slot = ensureSlot(CGF.getLoc(e->getSourceRange()), e->getType());
  CGF.buildCXXConstructExpr(e, slot);
}

void AggExprEmitter::VisitCompoundLiteralExpr(CompoundLiteralExpr *e) {
  if (dest.isPotentiallyAliased() && e->getType().isPODType(CGF.getContext())) {
    // For a POD type, just emit a load of the lvalue + a copy, because our
    // compound literal might alias the destination.
    buildAggLoadOfLValue(e);
    return;
  }

  AggValueSlot slot = ensureSlot(CGF.getLoc(e->getSourceRange()), e->getType());

  // Block-scope compound literals are destroyed at the end of the enclosing
  // scope in C.
  bool destruct =
      !CGF.getLangOpts().CPlusPlus && !slot.isExternallyDestructed();
  if (destruct)
    slot.setExternallyDestructed();

  CGF.buildAggExpr(e->getInitializer(), slot);

  if (destruct)
    if (QualType::DestructionKind dtorKind = e->getType().isDestructedType())
      llvm_unreachable("NYI");
}

void AggExprEmitter::VisitExprWithCleanups(ExprWithCleanups *e) {
  if (MissingFeatures::cleanups())
    llvm_unreachable("NYI");

  auto &builder = CGF.getBuilder();
  auto scopeLoc = CGF.getLoc(e->getSourceRange());
  mlir::OpBuilder::InsertPoint scopeBegin;
  builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        scopeBegin = b.saveInsertionPoint();
      });

  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(scopeBegin);
    CIRGenFunction::LexicalScope lexScope{CGF, scopeLoc,
                                          builder.getInsertionBlock()};
    Visit(e->getSubExpr());
  }
}

void AggExprEmitter::VisitLambdaExpr(LambdaExpr *e) {
  CIRGenFunction::SourceLocRAIIObject loc{CGF, CGF.getLoc(e->getSourceRange())};
  AggValueSlot slot = ensureSlot(CGF.getLoc(e->getSourceRange()), e->getType());
  LLVM_ATTRIBUTE_UNUSED LValue slotLv =
      CGF.makeAddrLValue(slot.getAddress(), e->getType());

  // We'll need to enter cleanup scopes in case any of the element
  // initializers throws an exception or contains branch out of the expressions.
  CIRGenFunction::CleanupDeactivationScope scope(CGF);

  auto curField = e->getLambdaClass()->field_begin();
  const auto *captureInfo = e->capture_begin();
  for (auto &captureInit : e->capture_inits()) {
    // Pick a name for the field.
    llvm::StringRef fieldName = curField->getName();
    const LambdaCapture &capture = *captureInfo;
    if (capture.capturesVariable()) {
      assert(!curField->isBitField() && "lambdas don't have bitfield members!");
      ValueDecl *v = capture.getCapturedVar();
      fieldName = v->getName();
      CGF.getCIRGenModule().LambdaFieldToName[*curField] = fieldName;
    } else {
      llvm_unreachable("NYI");
    }

    // Emit initialization
    LValue lv =
        CGF.buildLValueForFieldInitialization(slotLv, *curField, fieldName);
    if (curField->hasCapturedVLAType()) {
      llvm_unreachable("NYI");
    }

    buildInitializationToLValue(captureInit, lv);

    // Push a destructor if necessary.
    if (QualType::DestructionKind dtorKind =
            curField->getType().isDestructedType()) {
      llvm_unreachable("NYI");
    }

    curField++;
    captureInfo++;
  }
}

void AggExprEmitter::VisitCastExpr(CastExpr *e) {
  if (const auto *ece = dyn_cast<ExplicitCastExpr>(e))
    CGF.cgm.buildExplicitCastExprType(ece, &CGF);
  switch (e->getCastKind()) {
  case CK_LValueToRValueBitCast: {
    if (dest.isIgnored()) {
      CGF.buildAnyExpr(e->getSubExpr(), AggValueSlot::ignored(),
                       /*ignoreResult=*/true);
      break;
    }

    LValue sourceLv = CGF.buildLValue(e->getSubExpr());
    Address sourceAddress = sourceLv.getAddress();
    Address destAddress = dest.getAddress();

    auto loc = CGF.getLoc(e->getExprLoc());
    mlir::Value srcPtr = CGF.getBuilder().createBitcast(
        loc, sourceAddress.getPointer(), CGF.VoidPtrTy);
    mlir::Value dstPtr = CGF.getBuilder().createBitcast(
        loc, destAddress.getPointer(), CGF.VoidPtrTy);

    mlir::Value sizeVal = CGF.getBuilder().getConstInt(
        loc, CGF.SizeTy,
        CGF.getContext().getTypeSizeInChars(e->getType()).getQuantity());
    CGF.getBuilder().createMemCpy(loc, dstPtr, srcPtr, sizeVal);

    break;
  }

  case CK_ToUnion: {
    // Evaluate even if the destination is ignored.
    if (dest.isIgnored()) {
      CGF.buildAnyExpr(e->getSubExpr(), AggValueSlot::ignored(),
                       /*ignoreResult=*/true);
      break;
    }

    // GCC union extension
    QualType ty = e->getSubExpr()->getType();
    Address castPtr = dest.getAddress().withElementType(CGF.ConvertType(ty));
    buildInitializationToLValue(e->getSubExpr(),
                                CGF.makeAddrLValue(castPtr, ty));
    break;
  }

  case CK_LValueToRValue:
    // If we're loading from a volatile type, force the destination
    // into existence.
    if (e->getSubExpr()->getType().isVolatileQualified() ||
        MissingFeatures::volatileTypes()) {
      bool destruct =
          !dest.isExternallyDestructed() &&
          e->getType().isDestructedType() == QualType::DK_nontrivial_c_struct;
      if (destruct)
        dest.setExternallyDestructed();
      Visit(e->getSubExpr());

      if (destruct)
        CGF.pushDestroy(QualType::DK_nontrivial_c_struct, dest.getAddress(),
                        e->getType());

      return;
    }
    [[fallthrough]];

  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_ConstructorConversion:
    assert(CGF.getContext().hasSameUnqualifiedType(e->getSubExpr()->getType(),
                                                   e->getType()) &&
           "Implicit cast types must be compatible");
    Visit(e->getSubExpr());
    break;

  case CK_LValueBitCast:
    llvm_unreachable("should not be emitting lvalue bitcast as rvalue");

  case CK_Dependent:
  case CK_BitCast:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_ReinterpretMemberPointer:
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
  case CK_FloatingRealToComplex:
  case CK_FloatingComplexToReal:
  case CK_FloatingComplexToBoolean:
  case CK_FloatingComplexCast:
  case CK_FloatingComplexToIntegralComplex:
  case CK_IntegralRealToComplex:
  case CK_IntegralComplexToReal:
  case CK_IntegralComplexToBoolean:
  case CK_IntegralComplexCast:
  case CK_IntegralComplexToFloatingComplex:
  case CK_ARCProduceObject:
  case CK_ARCConsumeObject:
  case CK_ARCReclaimReturnedObject:
  case CK_ARCExtendBlockObject:
  case CK_CopyAndAutoreleaseBlockObject:
  case CK_BuiltinFnToFnPtr:
  case CK_ZeroToOCLOpaqueType:
  case CK_MatrixCast:

  case CK_IntToOCLSampler:
  case CK_FloatingToFixedPoint:
  case CK_FixedPointToFloating:
  case CK_FixedPointCast:
  case CK_FixedPointToBoolean:
  case CK_FixedPointToIntegral:
  case CK_IntegralToFixedPoint:
    llvm::errs() << "cast '" << e->getCastKindName()
                 << "' invalid for aggregate types\n";
    llvm_unreachable("cast kind invalid for aggregate types");
  default: {
    llvm::errs() << "cast kind not implemented: '" << e->getCastKindName()
                 << "'\n";
    assert(0 && "not implemented");
    break;
  }
  }
}

void AggExprEmitter::VisitCallExpr(const CallExpr *e) {
  if (e->getCallReturnType(CGF.getContext())->isReferenceType()) {
    llvm_unreachable("NYI");
  }

  withReturnValueSlot(
      e, [&](ReturnValueSlot slot) { return CGF.buildCallExpr(e, slot); });
}

void AggExprEmitter::withReturnValueSlot(
    const Expr *e, llvm::function_ref<RValue(ReturnValueSlot)> emitCall) {
  QualType retTy = e->getType();
  bool requiresDestruction =
      !dest.isExternallyDestructed() &&
      retTy.isDestructedType() == QualType::DK_nontrivial_c_struct;

  // If it makes no observable difference, save a memcpy + temporary.
  //
  // We need to always provide our own temporary if destruction is required.
  // Otherwise, EmitCall will emit its own, notice that it's "unused", and end
  // its lifetime before we have the chance to emit a proper destructor call.
  bool useTemp = dest.isPotentiallyAliased() || dest.requiresGCollection() ||
                 (requiresDestruction && !dest.getAddress().isValid());

  Address retAddr = Address::invalid();
  assert(!MissingFeatures::shouldEmitLifetimeMarkers() && "NYI");

  if (!useTemp) {
    retAddr = dest.getAddress();
  } else {
    retAddr = CGF.CreateMemTemp(retTy, CGF.getLoc(e->getSourceRange()), "tmp",
                                &retAddr);
    assert(!MissingFeatures::shouldEmitLifetimeMarkers() && "NYI");
  }

  RValue src =
      emitCall(ReturnValueSlot(retAddr, dest.isVolatile(), isResultUnused,
                               dest.isExternallyDestructed()));

  if (!useTemp)
    return;

  assert(dest.isIgnored() || dest.getPointer() != src.getAggregatePointer());
  buildFinalDestCopy(e->getType(), src);

  if (!requiresDestruction) {
    // If there's no dtor to run, the copy was the last use of our temporary.
    // Since we're not guaranteed to be in an ExprWithCleanups, clean up
    // eagerly.
    assert(!MissingFeatures::shouldEmitLifetimeMarkers() && "NYI");
  }
}

void AggExprEmitter::VisitBinCmp(const BinaryOperator *e) {
  assert(CGF.getContext().hasSameType(e->getLHS()->getType(),
                                      e->getRHS()->getType()));
  const ComparisonCategoryInfo &cmpInfo =
      CGF.getContext().CompCategories.getInfoForType(e->getType());
  assert(cmpInfo.Record->isTriviallyCopyable() &&
         "cannot copy non-trivially copyable aggregate");

  QualType argTy = e->getLHS()->getType();

  if (!argTy->isIntegralOrEnumerationType() && !argTy->isRealFloatingType() &&
      !argTy->isNullPtrType() && !argTy->isPointerType() &&
      !argTy->isMemberPointerType() && !argTy->isAnyComplexType())
    llvm_unreachable("aggregate three-way comparison");

  auto loc = CGF.getLoc(e->getSourceRange());

  if (e->getType()->isAnyComplexType())
    llvm_unreachable("NYI");

  auto lhs = CGF.buildAnyExpr(e->getLHS()).getScalarVal();
  auto rhs = CGF.buildAnyExpr(e->getRHS()).getScalarVal();

  mlir::Value resultScalar;
  if (argTy->isNullPtrType()) {
    resultScalar =
        CGF.builder.getConstInt(loc, cmpInfo.getEqualOrEquiv()->getIntValue());
  } else {
    auto ltRes = cmpInfo.getLess()->getIntValue();
    auto eqRes = cmpInfo.getEqualOrEquiv()->getIntValue();
    auto gtRes = cmpInfo.getGreater()->getIntValue();
    if (!cmpInfo.isPartial()) {
      // Strong ordering.
      resultScalar = CGF.builder.createThreeWayCmpStrong(loc, lhs, rhs, ltRes,
                                                         eqRes, gtRes);
    } else {
      // Partial ordering.
      auto unorderedRes = cmpInfo.getUnordered()->getIntValue();
      resultScalar = CGF.builder.createThreeWayCmpPartial(
          loc, lhs, rhs, ltRes, eqRes, gtRes, unorderedRes);
    }
  }

  // Create the return value in the destination slot.
  ensureDest(loc, e->getType());
  LValue destLv = CGF.makeAddrLValue(dest.getAddress(), e->getType());

  // Emit the address of the first (and only) field in the comparison category
  // type, and initialize it from the constant integer value produced above.
  const FieldDecl *resultField = *cmpInfo.Record->field_begin();
  LValue fieldLv = CGF.buildLValueForFieldInitialization(
      destLv, resultField, resultField->getName());
  CGF.buildStoreThroughLValue(RValue::get(resultScalar), fieldLv);

  // All done! The result is in the Dest slot.
}

void AggExprEmitter::VisitCXXParenListInitExpr(CXXParenListInitExpr *e) {
  visitCxxParenListOrInitListExpr(e, e->getInitExprs(),
                                  e->getInitializedFieldInUnion(),
                                  e->getArrayFiller());
}

void AggExprEmitter::VisitInitListExpr(InitListExpr *e) {
  // TODO(cir): use something like CGF.ErrorUnsupported
  if (e->hadArrayRangeDesignator())
    llvm_unreachable("GNU array range designator extension");

  if (e->isTransparent())
    return Visit(e->getInit(0));

  visitCxxParenListOrInitListExpr(
      e, e->inits(), e->getInitializedFieldInUnion(), e->getArrayFiller());
}

void AggExprEmitter::visitCxxParenListOrInitListExpr(
    Expr *ExprToVisit, ArrayRef<Expr *> initExprs,
    FieldDecl *initializedFieldInUnion, Expr *arrayFiller) {
#if 0
  // FIXME: Assess perf here?  Figure out what cases are worth optimizing here
  // (Length of globals? Chunks of zeroed-out space?).
  //
  // If we can, prefer a copy from a global; this is a lot less code for long
  // globals, and it's easier for the current optimizers to analyze.
  if (llvm::Constant *C =
          CGF.CGM.EmitConstantExpr(ExprToVisit, ExprToVisit->getType(), &CGF)) {
    llvm::GlobalVariable* GV =
    new llvm::GlobalVariable(CGF.CGM.getModule(), C->getType(), true,
                             llvm::GlobalValue::InternalLinkage, C, "");
    EmitFinalDestCopy(ExprToVisit->getType(),
                      CGF.MakeAddrLValue(GV, ExprToVisit->getType()));
    return;
  }
#endif

  AggValueSlot dest = ensureSlot(CGF.getLoc(ExprToVisit->getSourceRange()),
                                 ExprToVisit->getType());

  LValue destLv = CGF.makeAddrLValue(dest.getAddress(), ExprToVisit->getType());

  // Handle initialization of an array.
  if (ExprToVisit->getType()->isConstantArrayType()) {
    auto aType = cast<mlir::cir::ArrayType>(dest.getAddress().getElementType());
    buildArrayInit(dest.getAddress(), aType, ExprToVisit->getType(),
                   ExprToVisit, initExprs, arrayFiller);
    return;
  }
  if (ExprToVisit->getType()->isVariableArrayType()) {
    llvm_unreachable("variable arrays NYI");
    return;
  }

  if (ExprToVisit->getType()->isArrayType()) {
    llvm_unreachable("NYI");
  }

  assert(ExprToVisit->getType()->isRecordType() &&
         "Only support structs/unions here!");

  // Do struct initialization; this code just sets each individual member
  // to the approprate value.  This makes bitfield support automatic;
  // the disadvantage is that the generated code is more difficult for
  // the optimizer, especially with bitfields.
  unsigned numInitElements = initExprs.size();
  RecordDecl *record = ExprToVisit->getType()->castAs<RecordType>()->getDecl();

  // We'll need to enter cleanup scopes in case any of the element
  // initializers throws an exception.
  SmallVector<EHScopeStack::stable_iterator, 16> cleanups;
  CIRGenFunction::CleanupDeactivationScope deactivateCleanups(CGF);

  unsigned curInitIndex = 0;

  // Emit initialization of base classes.
  if (auto *cxxrd = dyn_cast<CXXRecordDecl>(record)) {
    assert(numInitElements >= cxxrd->getNumBases() &&
           "missing initializer for base class");
    for ([[maybe_unused]] auto &base : cxxrd->bases()) {
      llvm_unreachable("NYI");
    }
  }

  // Prepare a 'this' for CXXDefaultInitExprs.
  CIRGenFunction::FieldConstructionScope fcs(CGF, dest.getAddress());

  if (record->isUnion()) {
    // Only initialize one field of a union. The field itself is
    // specified by the initializer list.
    if (!initializedFieldInUnion) {
      // Empty union; we have nothing to do.

#ifndef NDEBUG
      // Make sure that it's really an empty and not a failure of
      // semantic analysis.
      for (const auto *field : record->fields())
        assert(
            (field->isUnnamedBitField() || field->isAnonymousStructOrUnion()) &&
            "Only unnamed bitfields or ananymous class allowed");
#endif
      return;
    }

    // FIXME: volatility
    FieldDecl *field = initializedFieldInUnion;

    LValue fieldLoc =
        CGF.buildLValueForFieldInitialization(destLv, field, field->getName());
    if (numInitElements) {
      // Store the initializer into the field
      buildInitializationToLValue(initExprs[0], fieldLoc);
    } else {
      // Default-initialize to null.
      buildNullInitializationToLValue(CGF.getLoc(ExprToVisit->getSourceRange()),
                                      fieldLoc);
    }

    return;
  }

  // Here we iterate over the fields; this makes it simpler to both
  // default-initialize fields and skip over unnamed fields.
  for (const auto *field : record->fields()) {
    // We're done once we hit the flexible array member.
    if (field->getType()->isIncompleteArrayType())
      break;

    // Always skip anonymous bitfields.
    if (field->isUnnamedBitField())
      continue;

    // We're done if we reach the end of the explicit initializers, we
    // have a zeroed object, and the rest of the fields are
    // zero-initializable.
    if (curInitIndex == numInitElements && dest.isZeroed() &&
        CGF.getTypes().isZeroInitializable(ExprToVisit->getType()))
      break;
    LValue lv =
        CGF.buildLValueForFieldInitialization(destLv, field, field->getName());
    // We never generate write-barries for initialized fields.
    assert(!MissingFeatures::setNonGC());

    if (curInitIndex < numInitElements) {
      // Store the initializer into the field.
      CIRGenFunction::SourceLocRAIIObject loc{
          CGF, CGF.getLoc(record->getSourceRange())};
      buildInitializationToLValue(initExprs[curInitIndex++], lv);
    } else {
      // We're out of initializers; default-initialize to null
      buildNullInitializationToLValue(CGF.getLoc(ExprToVisit->getSourceRange()),
                                      lv);
    }

    // Push a destructor if necessary.
    // FIXME: if we have an array of structures, all explicitly
    // initialized, we can end up pushing a linear number of cleanups.
    if (QualType::DestructionKind dtorKind =
            field->getType().isDestructedType()) {
      assert(lv.isSimple());
      if (dtorKind) {
        CGF.pushDestroyAndDeferDeactivation(NormalAndEHCleanup, lv.getAddress(),
                                            field->getType(),
                                            CGF.getDestroyer(dtorKind), false);
      }
    }

    // From LLVM codegen, maybe not useful for CIR:
    // If the GEP didn't get used because of a dead zero init or something
    // else, clean it up for -O0 builds and general tidiness.
  }
}

void AggExprEmitter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *e) {
  // Ensure that we have a slot, but if we already do, remember
  // whether it was externally destructed.
  bool wasExternallyDestructed = dest.isExternallyDestructed();
  ensureDest(CGF.getLoc(e->getSourceRange()), e->getType());

  // We're going to push a destructor if there isn't already one.
  dest.setExternallyDestructed();

  Visit(e->getSubExpr());

  // Push that destructor we promised.
  if (!wasExternallyDestructed)
    CGF.buildCXXTemporary(e->getTemporary(), e->getType(), dest.getAddress());
}

void AggExprEmitter::VisitAbstractConditionalOperator(
    const AbstractConditionalOperator *e) {
  auto &builder = CGF.getBuilder();
  auto loc = CGF.getLoc(e->getSourceRange());

  // Bind the common expression if necessary.
  CIRGenFunction::OpaqueValueMapping binding(CGF, e);
  CIRGenFunction::ConditionalEvaluation eval(CGF);
  assert(!MissingFeatures::getProfileCount());

  // Save whether the destination's lifetime is externally managed.
  bool isExternallyDestructed = dest.isExternallyDestructed();
  bool destructNonTrivialCStruct =
      !isExternallyDestructed &&
      e->getType().isDestructedType() == QualType::DK_nontrivial_c_struct;
  isExternallyDestructed |= destructNonTrivialCStruct;

  CGF.buildIfOnBoolExpr(
      e->getCond(), /*thenBuilder=*/
      [&](mlir::OpBuilder &, mlir::Location) {
        eval.begin(CGF);
        {
          CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                builder.getInsertionBlock()};
          dest.setExternallyDestructed(isExternallyDestructed);
          assert(!MissingFeatures::incrementProfileCounter());
          Visit(e->getTrueExpr());
        }
        eval.end(CGF);
      },
      loc,
      /*elseBuilder=*/
      [&](mlir::OpBuilder &, mlir::Location) {
        eval.begin(CGF);
        {
          CIRGenFunction::LexicalScope lexScope{CGF, loc,
                                                builder.getInsertionBlock()};
          // If the result of an agg expression is unused, then the emission
          // of the LHS might need to create a destination slot.  That's fine
          // with us, and we can safely emit the RHS into the same slot, but
          // we shouldn't claim that it's already being destructed.
          dest.setExternallyDestructed(isExternallyDestructed);
          assert(!MissingFeatures::incrementProfileCounter());
          Visit(e->getFalseExpr());
        }
        eval.end(CGF);
      },
      loc);

  if (destructNonTrivialCStruct)
    llvm_unreachable("NYI");
  assert(!MissingFeatures::incrementProfileCounter());
}

void AggExprEmitter::VisitBinComma(const BinaryOperator *e) {
  CGF.buildIgnoredExpr(e->getLHS());
  Visit(e->getRHS());
}

//===----------------------------------------------------------------------===//
//                        Helpers and dispatcher
//===----------------------------------------------------------------------===//

/// Get an approximate count of the number of non-zero bytes that will be stored
/// when outputting the initializer for the specified initializer expression.
/// FIXME(cir): this can be shared with LLVM codegen.
static CharUnits getNumNonZeroBytesInInit(const Expr *e, CIRGenFunction &cgf) {
  if (auto *mte = dyn_cast<MaterializeTemporaryExpr>(e))
    e = mte->getSubExpr();
  e = e->IgnoreParenNoopCasts(cgf.getContext());

  // 0 and 0.0 won't require any non-zero stores!
  if (isSimpleZero(e, cgf))
    return CharUnits::Zero();

  // If this is an initlist expr, sum up the size of sizes of the (present)
  // elements.  If this is something weird, assume the whole thing is non-zero.
  const InitListExpr *ile = dyn_cast<InitListExpr>(e);
  while (ile && ile->isTransparent())
    ile = dyn_cast<InitListExpr>(ile->getInit(0));
  if (!ile || !cgf.getTypes().isZeroInitializable(ile->getType()))
    return cgf.getContext().getTypeSizeInChars(e->getType());

  // InitListExprs for structs have to be handled carefully.  If there are
  // reference members, we need to consider the size of the reference, not the
  // referencee.  InitListExprs for unions and arrays can't have references.
  if (const RecordType *rt = e->getType()->getAs<RecordType>()) {
    if (!rt->isUnionType()) {
      RecordDecl *sd = rt->getDecl();
      CharUnits numNonZeroBytes = CharUnits::Zero();

      unsigned ileElement = 0;
      if (auto *cxxrd = dyn_cast<CXXRecordDecl>(sd))
        while (ileElement != cxxrd->getNumBases())
          numNonZeroBytes +=
              getNumNonZeroBytesInInit(ile->getInit(ileElement++), cgf);
      for (const auto *field : sd->fields()) {
        // We're done once we hit the flexible array member or run out of
        // InitListExpr elements.
        if (field->getType()->isIncompleteArrayType() ||
            ileElement == ile->getNumInits())
          break;
        if (field->isUnnamedBitField())
          continue;

        const Expr *e = ile->getInit(ileElement++);

        // Reference values are always non-null and have the width of a pointer.
        if (field->getType()->isReferenceType())
          numNonZeroBytes += cgf.getContext().toCharUnitsFromBits(
              cgf.getTarget().getPointerWidth(LangAS::Default));
        else
          numNonZeroBytes += getNumNonZeroBytesInInit(e, cgf);
      }

      return numNonZeroBytes;
    }
  }

  // FIXME: This overestimates the number of non-zero bytes for bit-fields.
  CharUnits numNonZeroBytes = CharUnits::Zero();
  for (unsigned i = 0, e = ile->getNumInits(); i != e; ++i)
    numNonZeroBytes += getNumNonZeroBytesInInit(ile->getInit(i), cgf);
  return numNonZeroBytes;
}

/// If the initializer is large and has a lot of zeros in it, emit a memset and
/// avoid storing the individual zeros.
static void checkAggExprForMemSetUse(AggValueSlot &slot, const Expr *e,
                                     CIRGenFunction &cgf) {
  // If the slot is arleady known to be zeroed, nothing to do. Don't mess with
  // volatile stores.
  if (slot.isZeroed() || slot.isVolatile() || !slot.getAddress().isValid())
    return;

  // C++ objects with a user-declared constructor don't need zero'ing.
  if (cgf.getLangOpts().CPlusPlus)
    if (const auto *rt = cgf.getContext()
                             .getBaseElementType(e->getType())
                             ->getAs<RecordType>()) {
      const auto *rd = cast<CXXRecordDecl>(rt->getDecl());
      if (rd->hasUserDeclaredConstructor())
        return;
    }

  // If the type is 16-bytes or smaller, prefer individual stores over memset.
  CharUnits size = slot.getPreferredSize(cgf.getContext(), e->getType());
  if (size <= CharUnits::fromQuantity(16))
    return;

  // Check to see if over 3/4 of the initializer are known to be zero.  If so,
  // we prefer to emit memset + individual stores for the rest.
  CharUnits numNonZeroBytes = getNumNonZeroBytesInInit(e, cgf);
  if (numNonZeroBytes * 4 > size)
    return;

  // Okay, it seems like a good idea to use an initial memset, emit the call.
  auto &builder = cgf.getBuilder();
  auto loc = cgf.getLoc(e->getSourceRange());
  Address slotAddr = slot.getAddress();
  auto zero = builder.getZero(loc, slotAddr.getElementType());

  builder.createStore(loc, zero, slotAddr);
  // Loc = CGF.Builder.CreateElementBitCast(Loc, CGF.Int8Ty);
  // CGF.Builder.CreateMemSet(Loc, CGF.Builder.getInt8(0), SizeVal, false);

  // Tell the AggExprEmitter that the slot is known zero.
  slot.setZeroed();
}

AggValueSlot::Overlap_t CIRGenFunction::getOverlapForBaseInit(
    const CXXRecordDecl *rd, const CXXRecordDecl *baseRd, bool isVirtual) {
  // If the most-derived object is a field declared with [[no_unique_address]],
  // the tail padding of any virtual base could be reused for other subobjects
  // of that field's class.
  if (isVirtual)
    return AggValueSlot::MayOverlap;

  // If the base class is laid out entirely within the nvsize of the derived
  // class, its tail padding cannot yet be initialized, so we can issue
  // stores at the full width of the base class.
  const ASTRecordLayout &layout = getContext().getASTRecordLayout(rd);
  if (layout.getBaseClassOffset(baseRd) +
          getContext().getASTRecordLayout(baseRd).getSize() <=
      layout.getNonVirtualSize())
    return AggValueSlot::DoesNotOverlap;

  // The tail padding may contain values we need to preserve.
  return AggValueSlot::MayOverlap;
}

void CIRGenFunction::buildAggExpr(const Expr *e, AggValueSlot slot) {
  assert(e && CIRGenFunction::hasAggregateEvaluationKind(e->getType()) &&
         "Invalid aggregate expression to emit");
  assert((slot.getAddress().isValid() || slot.isIgnored()) &&
         "slot has bits but no address");

  // Optimize the slot if possible.
  checkAggExprForMemSetUse(slot, e, *this);

  AggExprEmitter(*this, slot, slot.isIgnored()).Visit(const_cast<Expr *>(e));
}

void CIRGenFunction::buildAggregateCopy(LValue dest, LValue src, QualType ty,
                                        AggValueSlot::Overlap_t mayOverlap,
                                        bool isVolatile) {
  // TODO(cir): this function needs improvements, commented code for now since
  // this will be touched again soon.
  assert(!ty->isAnyComplexType() && "Shouldn't happen for complex");

  Address destPtr = dest.getAddress();
  Address srcPtr = src.getAddress();

  if (getLangOpts().CPlusPlus) {
    if (const RecordType *rt = ty->getAs<RecordType>()) {
      CXXRecordDecl *record = cast<CXXRecordDecl>(rt->getDecl());
      assert((record->hasTrivialCopyConstructor() ||
              record->hasTrivialCopyAssignment() ||
              record->hasTrivialMoveConstructor() ||
              record->hasTrivialMoveAssignment() ||
              record->hasAttr<TrivialABIAttr>() || record->isUnion()) &&
             "Trying to aggregate-copy a type without a trivial copy/move "
             "constructor or assignment operator");
      // Ignore empty classes in C++.
      if (record->isEmpty())
        return;
    }
  }

  if (getLangOpts().CUDAIsDevice) {
    llvm_unreachable("CUDA is NYI");
  }

  // Aggregate assignment turns into llvm.memcpy.  This is almost valid per
  // C99 6.5.16.1p3, which states "If the value being stored in an object is
  // read from another object that overlaps in anyway the storage of the first
  // object, then the overlap shall be exact and the two objects shall have
  // qualified or unqualified versions of a compatible type."
  //
  // memcpy is not defined if the source and destination pointers are exactly
  // equal, but other compilers do this optimization, and almost every memcpy
  // implementation handles this case safely.  If there is a libc that does not
  // safely handle this, we can add a target hook.

  // Get data size info for this aggregate. Don't copy the tail padding if this
  // might be a potentially-overlapping subobject, since the tail padding might
  // be occupied by a different object. Otherwise, copying it is fine.
  TypeInfoChars typeInfo;
  if (mayOverlap)
    typeInfo = getContext().getTypeInfoDataSizeInChars(ty);
  else
    typeInfo = getContext().getTypeInfoInChars(ty);

  mlir::Attribute sizeVal = nullptr;
  if (typeInfo.Width.isZero()) {
    // But note that getTypeInfo returns 0 for a VLA.
    if (auto *vat = dyn_cast_or_null<VariableArrayType>(
            getContext().getAsArrayType(ty))) {
      llvm_unreachable("VLA is NYI");
    }
  }
  if (!sizeVal) {
    // NOTE(cir): CIR types already carry info about their sizes. This is here
    // just for codegen parity.
    sizeVal = builder.getI64IntegerAttr(typeInfo.Width.getQuantity());
  }

  // FIXME: If we have a volatile struct, the optimizer can remove what might
  // appear to be `extra' memory ops:
  //
  // volatile struct { int i; } a, b;
  //
  // int main() {
  //   a = b;
  //   a = b;
  // }
  //
  // we need to use a different call here.  We use isVolatile to indicate when
  // either the source or the destination is volatile.

  // NOTE(cir): original codegen would normally convert DestPtr and SrcPtr to
  // i8* since memcpy operates on bytes. We don't need that in CIR because
  // cir.copy will operate on any CIR pointer that points to a sized type.

  // Don't do any of the memmove_collectable tests if GC isn't set.
  if (cgm.getLangOpts().getGC() == LangOptions::NonGC) {
    // fall through
  } else if (const RecordType *recordTy = ty->getAs<RecordType>()) {
    RecordDecl *record = recordTy->getDecl();
    if (record->hasObjectMember()) {
      llvm_unreachable("ObjC is NYI");
    }
  } else if (ty->isArrayType()) {
    QualType baseType = getContext().getBaseElementType(ty);
    if (const RecordType *recordTy = baseType->getAs<RecordType>()) {
      if (recordTy->getDecl()->hasObjectMember()) {
        llvm_unreachable("ObjC is NYI");
      }
    }
  }

  builder.createCopy(destPtr.getPointer(), srcPtr.getPointer(), isVolatile);

  // Determine the metadata to describe the position of any padding in this
  // memcpy, as well as the TBAA tags for the members of the struct, in case
  // the optimizer wishes to expand it in to scalar memory operations.
  if (cgm.getCodeGenOpts().NewStructPathTBAA || MissingFeatures::tbaa())
    llvm_unreachable("TBAA is NYI");
}

AggValueSlot::Overlap_t
CIRGenFunction::getOverlapForFieldInit(const FieldDecl *fd) {
  if (!fd->hasAttr<NoUniqueAddressAttr>() || !fd->getType()->isRecordType())
    return AggValueSlot::DoesNotOverlap;

  // If the field lies entirely within the enclosing class's nvsize, its tail
  // padding cannot overlap any already-initialized object. (The only subobjects
  // with greater addresses that might already be initialized are vbases.)
  const RecordDecl *classRd = fd->getParent();
  const ASTRecordLayout &layout = getContext().getASTRecordLayout(classRd);
  if (layout.getFieldOffset(fd->getFieldIndex()) +
          getContext().getTypeSize(fd->getType()) <=
      (uint64_t)getContext().toBits(layout.getNonVirtualSize()))
    return AggValueSlot::DoesNotOverlap;

  // The tail padding may contain values we need to preserve.
  return AggValueSlot::MayOverlap;
}

LValue CIRGenFunction::buildAggExprToLValue(const Expr *e) {
  assert(hasAggregateEvaluationKind(e->getType()) && "Invalid argument!");
  Address temp = CreateMemTemp(e->getType(), getLoc(e->getSourceRange()));
  LValue lv = makeAddrLValue(temp, e->getType());
  buildAggExpr(e, AggValueSlot::forLValue(lv, AggValueSlot::IsNotDestructed,
                                          AggValueSlot::DoesNotNeedGCBarriers,
                                          AggValueSlot::IsNotAliased,
                                          AggValueSlot::DoesNotOverlap));
  return lv;
}
